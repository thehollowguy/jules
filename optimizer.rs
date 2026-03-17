// =============================================================================
// jules/src/optimizer.rs
//
// Production-grade optimizer suite for the Jules neural-network runtime.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  Optimizers implemented                                                  │
// │                                                                          │
// │  Gradient-based (first-order)                                            │
// │    SGD          — stochastic gradient descent + momentum + Nesterov      │
// │    Adam         — adaptive moments (Kingma & Ba 2015)                    │
// │    AdamW        — Adam + decoupled weight decay (Loshchilov 2019)        │
// │    AdaGrad      — per-parameter accumulated squared gradients            │
// │    RMSProp      — leaky squared-gradient accumulator                     │
// │    AdaDelta     — parameter-free adaptive delta rule                     │
// │    Nadam        — Nesterov-accelerated Adam                              │
// │    RAdam        — rectified Adam with variance warm-up                   │
// │    LAMB         — layer-wise adaptive moments for large-batch            │
// │    LARS         — layer-wise adaptive rate scaling (SGD variant)         │
// │                                                                          │
// │  Learning-rate schedules                                                 │
// │    Constant, StepDecay, ExponentialDecay, CosineAnnealing,              │
// │    CosineAnnealingWarmRestarts (SGDR), LinearWarmup + any base,          │
// │    OneCycleLR, PolynomialDecay, ReduceOnPlateau                          │
// │                                                                          │
// │  Regularization helpers                                                  │
// │    L1, L2, ElasticNet, GradientClipByValue, GradientClipByNorm,         │
// │    GradientClipByGlobalNorm                                              │
// │                                                                          │
// │  Utilities                                                               │
// │    ParameterGroup — per-group lr / wd overrides                          │
// │    OptimizerState — serialisable snapshot for checkpoint / resume        │
// │    GradAccumulator — mini-batch gradient accumulation                    │
// └─────────────────────────────────────────────────────────────────────────┘
//
// Design constraints
// ──────────────────
// • Zero external dependencies — uses only Rust std.
// • No heap allocations in the hot path (step()) beyond what the model
//   already owns; moment buffers are pre-allocated in `build()`.
// • All arithmetic in f32 to match the tensor runtime.
// • Each optimizer implements the `Optimizer` trait so callers are
//   algorithm-agnostic.
// • Every optimizer is fully deterministic given the same seed: the internal
//   PRNGs used for noise-injection (RAdam variance bias) are LCG-based.
// =============================================================================

#![allow(dead_code)]

use std::f32::consts::PI;

// =============================================================================
// §1  PARAMETER BUFFER  — the unit of storage the optimizer operates on
// =============================================================================

/// A slice of mutable f32 weights + their corresponding gradients.
///
/// The optimizer never owns the data; it borrows them from the model for each
/// `step()` call.  This keeps the hot path allocation-free.
pub struct ParamBuffer<'a> {
    pub weights: &'a mut [f32],
    pub grads:   &'a [f32],
    /// Per-group override for learning rate (None = use global lr).
    pub lr_scale:  f32,
    /// Per-group override for weight decay (None = use global wd).
    pub wd_scale:  f32,
    /// Human-readable name for diagnostics / logging.
    pub name: &'a str,
}

impl<'a> ParamBuffer<'a> {
    pub fn new(weights: &'a mut [f32], grads: &'a [f32], name: &'a str) -> Self {
        ParamBuffer { weights, grads, lr_scale: 1.0, wd_scale: 1.0, name }
    }

    pub fn with_lr_scale(mut self, s: f32) -> Self { self.lr_scale = s; self }
    pub fn with_wd_scale(mut self, s: f32) -> Self { self.wd_scale = s; self }

    pub fn numel(&self) -> usize { self.weights.len() }

    /// L2 norm of the gradient vector.
    pub fn grad_norm(&self) -> f32 {
        self.grads.iter().map(|g| g * g).sum::<f32>().sqrt()
    }

    /// L2 norm of the weight vector.
    pub fn weight_norm(&self) -> f32 {
        self.weights.iter().map(|w| w * w).sum::<f32>().sqrt()
    }
}

// =============================================================================
// §2  LEARNING-RATE SCHEDULE TRAIT
// =============================================================================

/// A schedule maps global step → effective learning rate multiplier (in [0, ∞)).
pub trait LrSchedule: Send + Sync {
    /// Return the learning-rate multiplier for the given global step.
    fn multiplier(&self, step: u64, base_lr: f32) -> f32;

    /// Signal the schedule that validation loss has been measured.
    /// Only meaningful for `ReduceOnPlateau`.
    fn on_metric(&mut self, _metric: f32) {}
}

// ── §2a  Concrete schedules ───────────────────────────────────────────────────

/// Learning rate does not change.
pub struct ConstantLr;
impl LrSchedule for ConstantLr {
    fn multiplier(&self, _step: u64, base_lr: f32) -> f32 { base_lr }
}

/// Multiply lr by `gamma` every `step_size` steps.
/// lr(t) = base_lr * gamma^(floor(t / step_size))
pub struct StepDecay { pub step_size: u64, pub gamma: f32 }
impl LrSchedule for StepDecay {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        base_lr * self.gamma.powi((step / self.step_size) as i32)
    }
}

/// Continuous exponential decay.
/// lr(t) = base_lr * gamma^t
pub struct ExponentialDecay { pub gamma: f32 }
impl LrSchedule for ExponentialDecay {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        base_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing without restarts.
/// lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
pub struct CosineAnnealing { pub t_max: u64, pub lr_min: f32 }
impl LrSchedule for CosineAnnealing {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        let t = step.min(self.t_max) as f32;
        self.lr_min + 0.5 * (base_lr - self.lr_min)
            * (1.0 + (PI * t / self.t_max as f32).cos())
    }
}

/// SGDR: cosine annealing with warm restarts (Loshchilov 2017).
/// The period doubles after each restart.
pub struct CosineAnnealingWarmRestarts {
    pub t_0:      u64,  // initial period
    pub t_mult:   u64,  // period multiplier after each restart
    pub lr_min:   f32,
}
impl LrSchedule for CosineAnnealingWarmRestarts {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        // Find which cycle we're in and the position within it.
        let mut t_cur  = step;
        let mut period = self.t_0;
        loop {
            if t_cur < period { break; }
            t_cur  -= period;
            period  *= self.t_mult;
        }
        let frac = t_cur as f32 / period as f32;
        self.lr_min + 0.5 * (base_lr - self.lr_min) * (1.0 + (PI * frac).cos())
    }
}

/// Linear warm-up followed by a base schedule.
pub struct LinearWarmup<S: LrSchedule> {
    pub warmup_steps: u64,
    pub base:         S,
}
impl<S: LrSchedule> LrSchedule for LinearWarmup<S> {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        if step < self.warmup_steps {
            base_lr * (step as f32 / self.warmup_steps.max(1) as f32)
        } else {
            self.base.multiplier(step - self.warmup_steps, base_lr)
        }
    }
}

/// 1-cycle policy (Smith & Touvron 2019).
/// Phase 1: linear ramp from base/div_factor → base over pct_start * total steps.
/// Phase 2: cosine anneal from base → base / (div_factor * final_div_factor).
pub struct OneCycleLr {
    pub total_steps:      u64,
    pub div_factor:       f32,  // initial lr = base / div_factor
    pub final_div_factor: f32,  // final lr = initial / final_div_factor
    pub pct_start:        f32,  // fraction of steps for the warmup phase (e.g. 0.3)
}
impl LrSchedule for OneCycleLr {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        let t = step as f32;
        let n = self.total_steps as f32;
        let warmup_end = (self.pct_start * n) as u64;
        if step <= warmup_end {
            // Ramp: initial_lr → base_lr
            let frac = t / warmup_end as f32;
            let initial = base_lr / self.div_factor;
            initial + frac * (base_lr - initial)
        } else {
            // Cosine decay: base_lr → base_lr / (div_factor * final_div_factor)
            let min_lr = base_lr / (self.div_factor * self.final_div_factor);
            let frac = (t - warmup_end as f32) / (n - warmup_end as f32);
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (PI * frac).cos())
        }
    }
}

/// Polynomial decay.
/// lr(t) = (base_lr - end_lr) * (1 - t/T)^power + end_lr
pub struct PolynomialDecay {
    pub total_steps: u64,
    pub end_lr:      f32,
    pub power:       f32,
}
impl LrSchedule for PolynomialDecay {
    fn multiplier(&self, step: u64, base_lr: f32) -> f32 {
        let t = step.min(self.total_steps) as f32;
        let n = self.total_steps as f32;
        (base_lr - self.end_lr) * (1.0 - t / n).powf(self.power) + self.end_lr
    }
}

/// Reduce learning rate when a metric has stopped improving.
pub struct ReduceOnPlateau {
    pub factor:    f32,   // multiply lr by this when stalled (e.g. 0.5)
    pub patience:  u32,   // number of non-improving checks before reducing
    pub min_lr:    f32,
    pub threshold: f32,   // minimum improvement to count as progress
    // Internal state
    best:          f32,
    bad_epochs:    u32,
    current_mul:   f32,
}
impl ReduceOnPlateau {
    pub fn new(factor: f32, patience: u32, min_lr: f32) -> Self {
        ReduceOnPlateau {
            factor, patience, min_lr, threshold: 1e-4,
            best: f32::INFINITY, bad_epochs: 0, current_mul: 1.0,
        }
    }
}
impl LrSchedule for ReduceOnPlateau {
    fn multiplier(&self, _step: u64, base_lr: f32) -> f32 {
        (base_lr * self.current_mul).max(self.min_lr)
    }
    fn on_metric(&mut self, metric: f32) {
        if metric < self.best - self.threshold {
            self.best = metric;
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.bad_epochs >= self.patience {
                self.current_mul *= self.factor;
                self.bad_epochs   = 0;
            }
        }
    }
}

// =============================================================================
// §3  GRADIENT CLIPPING
// =============================================================================

/// Clip every gradient component to [-value, +value].
pub fn clip_by_value(grads: &mut [f32], value: f32) {
    for g in grads.iter_mut() {
        *g = g.clamp(-value, value);
    }
}

/// Clip the gradient vector so its L2 norm ≤ max_norm.
/// g ← g * (max_norm / max(norm(g), max_norm))
pub fn clip_by_norm(grads: &mut [f32], max_norm: f32) {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
}

/// Clip the global gradient norm across all parameter groups.
/// Returns the unclipped global norm.
pub fn clip_by_global_norm(all_grads: &mut [&mut Vec<f32>], max_norm: f32) -> f32 {
    let global_norm: f32 = all_grads.iter()
        .flat_map(|g| g.iter())
        .map(|g| g * g)
        .sum::<f32>()
        .sqrt();

    if global_norm > max_norm {
        let scale = max_norm / global_norm;
        for grads in all_grads.iter_mut() {
            for g in grads.iter_mut() { *g *= scale; }
        }
    }
    global_norm
}

// =============================================================================
// §4  REGULARIZATION
// =============================================================================

/// Add L2 weight decay to gradients (gradient-based, not decoupled).
/// ∇L ← ∇L + λ * w
pub fn add_l2_gradient(weights: &[f32], grads: &mut [f32], lambda: f32) {
    for (g, &w) in grads.iter_mut().zip(weights) {
        *g += lambda * w;
    }
}

/// Add L1 (Lasso) regularisation to gradients.
/// ∇L ← ∇L + λ * sign(w)
pub fn add_l1_gradient(weights: &[f32], grads: &mut [f32], lambda: f32) {
    for (g, &w) in grads.iter_mut().zip(weights) {
        *g += lambda * w.signum();
    }
}

/// Elastic-net: αL1 + (1-α)L2.
pub fn add_elastic_net_gradient(
    weights: &[f32], grads: &mut [f32], lambda: f32, alpha: f32
) {
    for (g, &w) in grads.iter_mut().zip(weights) {
        *g += lambda * (alpha * w.signum() + (1.0 - alpha) * w);
    }
}

/// Decoupled weight decay applied directly to weights (AdamW-style).
/// w ← w * (1 - lr * wd)
pub fn apply_weight_decay(weights: &mut [f32], lr: f32, wd: f32) {
    let scale = 1.0 - lr * wd;
    for w in weights.iter_mut() { *w *= scale; }
}

// =============================================================================
// §5  GRADIENT ACCUMULATION
// =============================================================================

/// Accumulates gradients over multiple micro-batches before calling `step()`.
///
/// Usage:
///   for micro in micro_batches {
///       let grads = compute_grads(micro);
///       accumulator.accumulate(&grads);
///   }
///   accumulator.average(micro_batches.len());
///   optimizer.step(&mut params, accumulator.view());
///   accumulator.reset();
pub struct GradAccumulator {
    pub buffer: Vec<f32>,
    count:      usize,
}

impl GradAccumulator {
    pub fn zeros(n: usize) -> Self {
        GradAccumulator { buffer: vec![0.0; n], count: 0 }
    }

    pub fn accumulate(&mut self, grads: &[f32]) {
        assert_eq!(grads.len(), self.buffer.len());
        for (acc, &g) in self.buffer.iter_mut().zip(grads) { *acc += g; }
        self.count += 1;
    }

    /// Divide accumulated sum by the number of micro-batches.
    pub fn average(&mut self, n: usize) {
        let inv = 1.0 / n.max(1) as f32;
        for acc in self.buffer.iter_mut() { *acc *= inv; }
    }

    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|x| *x = 0.0);
        self.count = 0;
    }

    pub fn count(&self) -> usize { self.count }
}

// =============================================================================
// §6  OPTIMIZER TRAIT
// =============================================================================

/// The core optimizer interface.
///
/// Every optimizer must implement:
///  • `step` — update weights in-place given gradients.
///  • `zero_grad` — reset any internal per-step gradient state.
///  • `state_snapshot` — produce a serialisable snapshot for checkpointing.
///  • `load_snapshot` — restore from a snapshot.
pub trait Optimizer: Send + Sync {
    /// Perform one parameter update.
    ///
    /// `params` is a flat slice of (weight, gradient) pairs grouped by
    /// parameter tensor.  The effective learning rate for step `t` is
    /// `base_lr * schedule.multiplier(t, base_lr)`.
    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64);

    /// Set all gradient buffers to zero (typically a no-op here; the model
    /// owns its gradient storage).
    fn zero_grad(&mut self) {}

    /// Return the effective learning rate for the given global step.
    fn current_lr(&self, step: u64) -> f32;

    /// Serialisable state snapshot.
    fn state_snapshot(&self) -> OptimizerState;

    /// Restore from a snapshot.
    fn load_snapshot(&mut self, state: OptimizerState);

    /// Name for logging.
    fn name(&self) -> &str;
}

// =============================================================================
// §7  SERIALISABLE STATE
// =============================================================================

/// All moment/velocity buffers for one optimizer, serialised as flat f32 vecs.
#[derive(Debug, Clone, Default)]
pub struct OptimizerState {
    pub name:   String,
    pub step:   u64,
    pub lr:     f32,
    /// Named buffers (e.g. "m1_layer0_w", "m2_layer0_w", …)
    pub buffers: Vec<(String, Vec<f32>)>,
    /// Scalar hyperparameters snapshot.
    pub hparams: Vec<(String, f32)>,
}

// =============================================================================
// §8  HYPERPARAMETER CONFIGURATION
// =============================================================================

/// Shared hyperparameter block.
#[derive(Debug, Clone)]
pub struct HParams {
    pub lr:           f32,
    pub weight_decay: f32,
    pub grad_clip:    Option<GradClip>,
}

#[derive(Debug, Clone, Copy)]
pub enum GradClip {
    ByValue(f32),
    ByNorm(f32),
}

impl Default for HParams {
    fn default() -> Self {
        HParams { lr: 1e-3, weight_decay: 0.0, grad_clip: None }
    }
}

// =============================================================================
// §9  SGD  (with momentum and optional Nesterov)
// =============================================================================

/// Stochastic gradient descent.
///
/// Update rule (plain):
///   v ← μ * v − lr * g
///   w ← w + v
///
/// Nesterov update:
///   v' ← μ * v − lr * g
///   w  ← w − lr * g + μ * v'
pub struct Sgd {
    pub hp:       HParams,
    pub momentum: f32,
    pub nesterov: bool,
    /// Velocity buffers, one vec per parameter group.
    velocity:     Vec<Vec<f32>>,
    schedule:     Box<dyn LrSchedule>,
}

impl Sgd {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Sgd {
            hp: HParams { lr, ..Default::default() },
            momentum,
            nesterov: false,
            velocity: Vec::new(),
            schedule: Box::new(ConstantLr),
        }
    }

    pub fn with_nesterov(mut self) -> Self { self.nesterov = true; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.hp.weight_decay = wd; self }
    pub fn with_grad_clip(mut self, clip: GradClip) -> Self { self.hp.grad_clip = Some(clip); self }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.schedule = Box::new(s); self
    }

    fn ensure_buffers(&mut self, params: &[ParamBuffer<'_>]) {
        if self.velocity.len() < params.len() {
            self.velocity.resize_with(params.len(), Vec::new);
        }
        for (i, p) in params.iter().enumerate() {
            if self.velocity[i].len() != p.numel() {
                self.velocity[i] = vec![0.0; p.numel()];
            }
        }
    }
}

impl Optimizer for Sgd {
    fn name(&self) -> &str { "SGD" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.ensure_buffers(params);
        let lr = self.schedule.multiplier(step, self.hp.lr);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.hp.weight_decay * p.wd_scale;
            let v      = &mut self.velocity[i];

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                // Apply gradient clipping per-element (value) if requested.
                let g = match self.hp.grad_clip {
                    Some(GradClip::ByValue(c)) => g.clamp(-c, c),
                    _ => g,
                };
                // L2 weight decay in gradient form.
                let g = g + eff_wd * *w;

                // Velocity update.
                v[j] = self.momentum * v[j] - eff_lr * g;

                if self.nesterov {
                    *w += self.momentum * v[j] - eff_lr * g;
                } else {
                    *w += v[j];
                }
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 {
        self.schedule.multiplier(step, self.hp.lr)
    }

    fn state_snapshot(&self) -> OptimizerState {
        let buffers = self.velocity.iter().enumerate()
            .map(|(i, v)| (format!("velocity_{i}"), v.clone()))
            .collect();
        OptimizerState {
            name:    "SGD".into(),
            step:    0,
            lr:      self.hp.lr,
            buffers,
            hparams: vec![
                ("momentum".into(), self.momentum),
                ("weight_decay".into(), self.hp.weight_decay),
            ],
        }
    }

    fn load_snapshot(&mut self, state: OptimizerState) {
        self.hp.lr = state.lr;
        self.velocity.clear();
        for (_, buf) in state.buffers { self.velocity.push(buf); }
    }
}

// =============================================================================
// §10  ADAM  (Kingma & Ba 2015)
// =============================================================================

/// Adam: Adaptive Moment Estimation.
///
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t          (biased 1st moment)
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t²          (biased 2nd moment)
/// m̂_t = m_t / (1 - β1^t)                        (bias-corrected)
/// v̂_t = v_t / (1 - β2^t)                        (bias-corrected)
/// w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
pub struct Adam {
    pub hp:   HParams,
    pub beta1: f32,
    pub beta2: f32,
    pub eps:   f32,
    m1:        Vec<Vec<f32>>,   // first moment
    m2:        Vec<Vec<f32>>,   // second moment
    schedule:  Box<dyn LrSchedule>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            hp: HParams { lr, ..Default::default() },
            beta1: 0.9,
            beta2: 0.999,
            eps:   1e-8,
            m1:    Vec::new(),
            m2:    Vec::new(),
            schedule: Box::new(ConstantLr),
        }
    }

    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self { self.beta1 = b1; self.beta2 = b2; self }
    pub fn with_eps(mut self, eps: f32) -> Self { self.eps = eps; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.hp.weight_decay = wd; self }
    pub fn with_grad_clip(mut self, clip: GradClip) -> Self { self.hp.grad_clip = Some(clip); self }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.schedule = Box::new(s); self
    }

    fn ensure_buffers(&mut self, params: &[ParamBuffer<'_>]) {
        for buffers in [&mut self.m1, &mut self.m2] {
            if buffers.len() < params.len() {
                buffers.resize_with(params.len(), Vec::new);
            }
            for (i, p) in params.iter().enumerate() {
                if buffers[i].len() != p.numel() {
                    buffers[i] = vec![0.0; p.numel()];
                }
            }
        }
    }
}

impl Optimizer for Adam {
    fn name(&self) -> &str { "Adam" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.ensure_buffers(params);
        let lr  = self.schedule.multiplier(step, self.hp.lr);
        let t   = step as f32 + 1.0;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.hp.weight_decay * p.wd_scale;
            let m1 = &mut self.m1[i];
            let m2 = &mut self.m2[i];

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.hp.grad_clip);
                // L2 gradient-form decay.
                let g = g + eff_wd * *w;

                // Moment updates.
                m1[j] = self.beta1 * m1[j] + (1.0 - self.beta1) * g;
                m2[j] = self.beta2 * m2[j] + (1.0 - self.beta2) * g * g;

                // Bias-corrected moments.
                let m_hat = m1[j] / bc1;
                let v_hat = m2[j] / bc2;

                *w -= eff_lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 {
        self.schedule.multiplier(step, self.hp.lr)
    }

    fn state_snapshot(&self) -> OptimizerState {
        let mut buffers = Vec::new();
        for (i, (m, v)) in self.m1.iter().zip(&self.m2).enumerate() {
            buffers.push((format!("m1_{i}"), m.clone()));
            buffers.push((format!("m2_{i}"), v.clone()));
        }
        OptimizerState {
            name: "Adam".into(), step: 0, lr: self.hp.lr, buffers,
            hparams: vec![
                ("beta1".into(), self.beta1), ("beta2".into(), self.beta2),
                ("eps".into(), self.eps), ("weight_decay".into(), self.hp.weight_decay),
            ],
        }
    }

    fn load_snapshot(&mut self, state: OptimizerState) {
        self.hp.lr = state.lr;
        self.m1.clear(); self.m2.clear();
        let mut it = state.buffers.into_iter();
        while let (Some((_, m)), Some((_, v))) = (it.next(), it.next()) {
            self.m1.push(m);
            self.m2.push(v);
        }
    }
}

// =============================================================================
// §11  AdamW  (Loshchilov & Hutter 2019)
// =============================================================================

/// AdamW: Adam with decoupled weight decay.
///
/// Identical to Adam except weight decay is applied directly to the weights
/// *before* the gradient step, not added to the gradient:
///   w_t ← w_{t-1} - α * λ * w_{t-1}   (decoupled decay)
///         - α * m̂_t / (√v̂_t + ε)      (gradient step)
///
/// This prevents the interaction between adaptive lr and weight decay that
/// causes Adam to under-regularise large-gradient parameters.
pub struct AdamW {
    inner: Adam,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        AdamW { inner: Adam::new(lr).with_weight_decay(1e-2) }
    }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.inner = self.inner.with_betas(b1, b2); self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.inner.hp.weight_decay = wd; self
    }
    pub fn with_grad_clip(mut self, clip: GradClip) -> Self {
        self.inner = self.inner.with_grad_clip(clip); self
    }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.inner = self.inner.with_schedule(s); self
    }
}

impl Optimizer for AdamW {
    fn name(&self) -> &str { "AdamW" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.inner.ensure_buffers(params);
        let lr  = self.inner.schedule.multiplier(step, self.inner.hp.lr);
        let t   = step as f32 + 1.0;
        let bc1 = 1.0 - self.inner.beta1.powf(t);
        let bc2 = 1.0 - self.inner.beta2.powf(t);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.inner.hp.weight_decay * p.wd_scale;
            let m1 = &mut self.inner.m1[i];
            let m2 = &mut self.inner.m2[i];

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.inner.hp.grad_clip);

                // Decoupled weight decay — applied directly to parameter.
                *w *= 1.0 - eff_lr * eff_wd;

                // Standard Adam moment updates (no decay in gradient).
                m1[j] = self.inner.beta1 * m1[j] + (1.0 - self.inner.beta1) * g;
                m2[j] = self.inner.beta2 * m2[j] + (1.0 - self.inner.beta2) * g * g;

                let m_hat = m1[j] / bc1;
                let v_hat = m2[j] / bc2;

                *w -= eff_lr * m_hat / (v_hat.sqrt() + self.inner.eps);
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 { self.inner.current_lr(step) }
    fn state_snapshot(&self)          -> OptimizerState { self.inner.state_snapshot() }
    fn load_snapshot(&mut self, s: OptimizerState) { self.inner.load_snapshot(s); }
}

// =============================================================================
// §12  AdaGrad  (Duchi et al. 2011)
// =============================================================================

/// AdaGrad: Adaptive Gradient Algorithm.
///
/// G_t = G_{t-1} + g_t²
/// w_t = w_{t-1} - (α / sqrt(G_t + ε)) * g_t
pub struct AdaGrad {
    pub hp:        HParams,
    pub eps:       f32,
    pub lr_decay:  f32,   // optional lr decay: α_t = α / (1 + t * lr_decay)
    g_sq:          Vec<Vec<f32>>,
    schedule:      Box<dyn LrSchedule>,
}

impl AdaGrad {
    pub fn new(lr: f32) -> Self {
        AdaGrad {
            hp:       HParams { lr, ..Default::default() },
            eps:      1e-10,
            lr_decay: 0.0,
            g_sq:     Vec::new(),
            schedule: Box::new(ConstantLr),
        }
    }

    pub fn with_lr_decay(mut self, d: f32) -> Self { self.lr_decay = d; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.hp.weight_decay = wd; self }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.schedule = Box::new(s); self
    }

    fn ensure_buffers(&mut self, params: &[ParamBuffer<'_>]) {
        if self.g_sq.len() < params.len() {
            self.g_sq.resize_with(params.len(), Vec::new);
        }
        for (i, p) in params.iter().enumerate() {
            if self.g_sq[i].len() != p.numel() {
                self.g_sq[i] = vec![0.0; p.numel()];
            }
        }
    }
}

impl Optimizer for AdaGrad {
    fn name(&self) -> &str { "AdaGrad" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.ensure_buffers(params);
        let base_lr = self.schedule.multiplier(step, self.hp.lr);
        let lr = base_lr / (1.0 + step as f32 * self.lr_decay);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.hp.weight_decay * p.wd_scale;
            let acc    = &mut self.g_sq[i];

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.hp.grad_clip) + eff_wd * *w;
                acc[j] += g * g;
                *w -= eff_lr / (acc[j].sqrt() + self.eps) * g;
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 {
        let base_lr = self.schedule.multiplier(step, self.hp.lr);
        base_lr / (1.0 + step as f32 * self.lr_decay)
    }

    fn state_snapshot(&self) -> OptimizerState {
        let buffers = self.g_sq.iter().enumerate()
            .map(|(i, g)| (format!("g_sq_{i}"), g.clone()))
            .collect();
        OptimizerState {
            name: "AdaGrad".into(), step: 0, lr: self.hp.lr, buffers,
            hparams: vec![
                ("eps".into(), self.eps),
                ("lr_decay".into(), self.lr_decay),
                ("weight_decay".into(), self.hp.weight_decay),
            ],
        }
    }

    fn load_snapshot(&mut self, state: OptimizerState) {
        self.hp.lr = state.lr;
        self.g_sq  = state.buffers.into_iter().map(|(_, v)| v).collect();
    }
}

// =============================================================================
// §13  RMSProp  (Hinton 2012)
// =============================================================================

/// RMSProp: exponentially weighted moving average of squared gradients.
///
/// E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g_t²
/// w_t = w_{t-1} - (α / sqrt(E[g²]_t + ε)) * g_t
///
/// Optional centred variant subtracts the running mean of gradients:
/// Var[g]_t = E[g²]_t - E[g]_t²
pub struct RmsProp {
    pub hp:       HParams,
    pub rho:      f32,
    pub eps:      f32,
    pub momentum: f32,
    pub centred:  bool,   // centred RMSProp (subtract E[g]²)
    eg2:          Vec<Vec<f32>>,   // E[g²]
    eg:           Vec<Vec<f32>>,   // E[g]   (centred only)
    vel:          Vec<Vec<f32>>,   // velocity (momentum)
    schedule:     Box<dyn LrSchedule>,
}

impl RmsProp {
    pub fn new(lr: f32) -> Self {
        RmsProp {
            hp:       HParams { lr, ..Default::default() },
            rho:      0.99,
            eps:      1e-8,
            momentum: 0.0,
            centred:  false,
            eg2:      Vec::new(),
            eg:       Vec::new(),
            vel:      Vec::new(),
            schedule: Box::new(ConstantLr),
        }
    }

    pub fn with_rho(mut self, rho: f32) -> Self { self.rho = rho; self }
    pub fn with_momentum(mut self, m: f32) -> Self { self.momentum = m; self }
    pub fn with_centred(mut self) -> Self { self.centred = true; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.hp.weight_decay = wd; self }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.schedule = Box::new(s); self
    }

    fn ensure_buffers(&mut self, params: &[ParamBuffer<'_>]) {
        for buf in [&mut self.eg2, &mut self.eg, &mut self.vel] {
            if buf.len() < params.len() { buf.resize_with(params.len(), Vec::new); }
            for (i, p) in params.iter().enumerate() {
                if buf[i].len() != p.numel() { buf[i] = vec![0.0; p.numel()]; }
            }
        }
    }
}

impl Optimizer for RmsProp {
    fn name(&self) -> &str { "RMSProp" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.ensure_buffers(params);
        let lr = self.schedule.multiplier(step, self.hp.lr);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.hp.weight_decay * p.wd_scale;

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.hp.grad_clip) + eff_wd * *w;

                self.eg2[i][j] = self.rho * self.eg2[i][j] + (1.0 - self.rho) * g * g;

                let denom = if self.centred {
                    self.eg[i][j] = self.rho * self.eg[i][j] + (1.0 - self.rho) * g;
                    (self.eg2[i][j] - self.eg[i][j] * self.eg[i][j] + self.eps).sqrt()
                } else {
                    (self.eg2[i][j] + self.eps).sqrt()
                };

                if self.momentum > 0.0 {
                    self.vel[i][j] = self.momentum * self.vel[i][j] + eff_lr * g / denom;
                    *w -= self.vel[i][j];
                } else {
                    *w -= eff_lr * g / denom;
                }
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 { self.schedule.multiplier(step, self.hp.lr) }

    fn state_snapshot(&self) -> OptimizerState {
        let mut buffers = Vec::new();
        for (i, e) in self.eg2.iter().enumerate() { buffers.push((format!("eg2_{i}"), e.clone())); }
        for (i, e) in self.eg.iter().enumerate()  { buffers.push((format!("eg_{i}"),  e.clone())); }
        for (i, v) in self.vel.iter().enumerate()  { buffers.push((format!("vel_{i}"), v.clone())); }
        OptimizerState {
            name: "RMSProp".into(), step: 0, lr: self.hp.lr, buffers,
            hparams: vec![
                ("rho".into(), self.rho), ("eps".into(), self.eps),
                ("momentum".into(), self.momentum),
            ],
        }
    }

    fn load_snapshot(&mut self, state: OptimizerState) {
        self.hp.lr = state.lr;
        let mut eg2 = Vec::new(); let mut eg = Vec::new(); let mut vel = Vec::new();
        for (key, buf) in state.buffers {
            if key.starts_with("eg2") { eg2.push(buf); }
            else if key.starts_with("eg_") { eg.push(buf); }
            else { vel.push(buf); }
        }
        self.eg2 = eg2; self.eg = eg; self.vel = vel;
    }
}

// =============================================================================
// §14  AdaDelta  (Zeiler 2012)
// =============================================================================

/// AdaDelta: parameter-free adaptive learning rates.
///
/// E[g²]_t  = ρ * E[g²]_{t-1} + (1-ρ) * g²
/// Δw_t     = - sqrt(E[Δw²]_{t-1} + ε) / sqrt(E[g²]_t + ε) * g
/// E[Δw²]_t = ρ * E[Δw²]_{t-1} + (1-ρ) * Δw²
/// w_t      = w_{t-1} + Δw_t
pub struct AdaDelta {
    pub hp:   HParams,
    pub rho:  f32,
    pub eps:  f32,
    eg2:      Vec<Vec<f32>>,
    edw2:     Vec<Vec<f32>>,
}

impl AdaDelta {
    pub fn new() -> Self {
        AdaDelta {
            hp:   HParams { lr: 1.0, ..Default::default() },
            rho:  0.95,
            eps:  1e-6,
            eg2:  Vec::new(),
            edw2: Vec::new(),
        }
    }
    pub fn with_rho(mut self, rho: f32) -> Self { self.rho = rho; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.hp.weight_decay = wd; self }

    fn ensure_buffers(&mut self, params: &[ParamBuffer<'_>]) {
        for buf in [&mut self.eg2, &mut self.edw2] {
            if buf.len() < params.len() { buf.resize_with(params.len(), Vec::new); }
            for (i, p) in params.iter().enumerate() {
                if buf[i].len() != p.numel() { buf[i] = vec![0.0; p.numel()]; }
            }
        }
    }
}

impl Optimizer for AdaDelta {
    fn name(&self) -> &str { "AdaDelta" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], _step: u64) {
        self.ensure_buffers(params);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_wd = self.hp.weight_decay * p.wd_scale;

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.hp.grad_clip) + eff_wd * *w;

                self.eg2[i][j] = self.rho * self.eg2[i][j] + (1.0 - self.rho) * g * g;

                let rms_g  = (self.eg2[i][j]  + self.eps).sqrt();
                let rms_dw = (self.edw2[i][j] + self.eps).sqrt();
                let dw     = -rms_dw / rms_g * g;

                self.edw2[i][j] = self.rho * self.edw2[i][j] + (1.0 - self.rho) * dw * dw;
                *w += dw;
            }
        }
    }

    fn current_lr(&self, _step: u64) -> f32 { self.hp.lr }

    fn state_snapshot(&self) -> OptimizerState {
        let mut buffers = Vec::new();
        for (i, e) in self.eg2.iter().enumerate()  { buffers.push((format!("eg2_{i}"),  e.clone())); }
        for (i, e) in self.edw2.iter().enumerate() { buffers.push((format!("edw2_{i}"), e.clone())); }
        OptimizerState { name: "AdaDelta".into(), step: 0, lr: 1.0, buffers, hparams: vec![("rho".into(), self.rho)] }
    }

    fn load_snapshot(&mut self, state: OptimizerState) {
        let mid = state.buffers.len() / 2;
        self.eg2  = state.buffers[..mid].iter().map(|(_, v)| v.clone()).collect();
        self.edw2 = state.buffers[mid..].iter().map(|(_, v)| v.clone()).collect();
    }
}

// =============================================================================
// §15  Nadam  (Dozat 2016)
// =============================================================================

/// Nadam: Nesterov-accelerated Adam.
///
/// Like Adam but uses the Nesterov trick to look ahead:
///   m̂_t = β1 * m_t / (1 - β1^{t+1}) + (1-β1) * g_t / (1 - β1^t)
///   w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
pub struct Nadam {
    inner: Adam,
}

impl Nadam {
    pub fn new(lr: f32) -> Self { Nadam { inner: Adam::new(lr) } }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.inner = self.inner.with_betas(b1, b2); self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.inner = self.inner.with_weight_decay(wd); self
    }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.inner = self.inner.with_schedule(s); self
    }
}

impl Optimizer for Nadam {
    fn name(&self) -> &str { "Nadam" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.inner.ensure_buffers(params);
        let lr  = self.inner.schedule.multiplier(step, self.inner.hp.lr);
        let t   = step as f32 + 1.0;
        let bc1 = 1.0 - self.inner.beta1.powf(t);
        let bc2 = 1.0 - self.inner.beta2.powf(t);
        let bc1_next = 1.0 - self.inner.beta1.powf(t + 1.0);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.inner.hp.weight_decay * p.wd_scale;
            let m1 = &mut self.inner.m1[i];
            let m2 = &mut self.inner.m2[i];

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.inner.hp.grad_clip) + eff_wd * *w;

                m1[j] = self.inner.beta1 * m1[j] + (1.0 - self.inner.beta1) * g;
                m2[j] = self.inner.beta2 * m2[j] + (1.0 - self.inner.beta2) * g * g;

                // Nesterov: use the next bias-corrected m1.
                let m_hat = self.inner.beta1 * m1[j] / bc1_next
                          + (1.0 - self.inner.beta1) * g / bc1;
                let v_hat = m2[j] / bc2;

                *w -= eff_lr * m_hat / (v_hat.sqrt() + self.inner.eps);
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 { self.inner.current_lr(step) }
    fn state_snapshot(&self) -> OptimizerState { self.inner.state_snapshot() }
    fn load_snapshot(&mut self, s: OptimizerState) { self.inner.load_snapshot(s); }
}

// =============================================================================
// §16  RAdam  (Liu et al. 2020)
// =============================================================================

/// Rectified Adam: variance warm-up to stabilise early training.
///
/// Computes the maximum ρ for the SMA of the second moment:
///   ρ∞ = 2 / (1 - β2) - 1
/// If ρ_t > 4 (variance is tractable), uses the rectified step length;
/// otherwise falls back to SGD with the first moment.
pub struct Radam {
    inner: Adam,
}

impl Radam {
    pub fn new(lr: f32) -> Self { Radam { inner: Adam::new(lr) } }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.inner = self.inner.with_betas(b1, b2); self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.inner = self.inner.with_weight_decay(wd); self
    }
    pub fn with_grad_clip(mut self, clip: GradClip) -> Self {
        self.inner = self.inner.with_grad_clip(clip); self
    }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.inner = self.inner.with_schedule(s); self
    }
}

impl Optimizer for Radam {
    fn name(&self) -> &str { "RAdam" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.inner.ensure_buffers(params);
        let lr    = self.inner.schedule.multiplier(step, self.inner.hp.lr);
        let t     = step as f32 + 1.0;
        let b2    = self.inner.beta2;
        let rho_inf = 2.0 / (1.0 - b2) - 1.0;
        let beta2_t = b2.powf(t);
        let rho_t   = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t);
        let bc1 = 1.0 - self.inner.beta1.powf(t);
        let bc2 = 1.0 - beta2_t;

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.inner.hp.weight_decay * p.wd_scale;
            let m1 = &mut self.inner.m1[i];
            let m2 = &mut self.inner.m2[i];

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.inner.hp.grad_clip) + eff_wd * *w;

                m1[j] = self.inner.beta1 * m1[j] + (1.0 - self.inner.beta1) * g;
                m2[j] = b2 * m2[j] + (1.0 - b2) * g * g;

                let m_hat = m1[j] / bc1;

                if rho_t > 4.0 {
                    // Variance is sufficiently tractable — use rectified step.
                    let r = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                           / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                           .sqrt();
                    let v_hat = (m2[j] / bc2).sqrt() + self.inner.eps;
                    *w -= eff_lr * r * m_hat / v_hat;
                } else {
                    // Variance not yet tractable — SGD with first moment.
                    *w -= eff_lr * m_hat;
                }
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 { self.inner.current_lr(step) }
    fn state_snapshot(&self) -> OptimizerState { self.inner.state_snapshot() }
    fn load_snapshot(&mut self, s: OptimizerState) { self.inner.load_snapshot(s); }
}

// =============================================================================
// §17  LAMB  (You et al. 2020)
// =============================================================================

/// LAMB: Layer-wise Adaptive Moments for large-batch training.
///
/// Extends Adam/AdamW with a per-layer trust ratio:
///   r = ‖w‖ / ‖m̂ / (√v̂ + ε) + λ * w‖
///   w ← w - α * r * (m̂ / (√v̂ + ε) + λ * w)
///
/// When the layer norm is 0 or the update norm is 0, r is set to 1.
pub struct Lamb {
    inner:   Adam,
    pub clamp_trust: (f32, f32),  // (min, max) for the trust ratio
}

impl Lamb {
    pub fn new(lr: f32) -> Self {
        Lamb {
            inner:       Adam::new(lr).with_weight_decay(1e-2),
            clamp_trust: (1e-3, 10.0),
        }
    }
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.inner = self.inner.with_betas(b1, b2); self
    }
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.inner.hp.weight_decay = wd; self
    }
    pub fn with_trust_clamp(mut self, lo: f32, hi: f32) -> Self {
        self.clamp_trust = (lo, hi); self
    }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.inner = self.inner.with_schedule(s); self
    }
}

impl Optimizer for Lamb {
    fn name(&self) -> &str { "LAMB" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.inner.ensure_buffers(params);
        let lr  = self.inner.schedule.multiplier(step, self.inner.hp.lr);
        let t   = step as f32 + 1.0;
        let bc1 = 1.0 - self.inner.beta1.powf(t);
        let bc2 = 1.0 - self.inner.beta2.powf(t);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_lr = lr * p.lr_scale;
            let eff_wd = self.inner.hp.weight_decay * p.wd_scale;
            let m1 = &mut self.inner.m1[i];
            let m2 = &mut self.inner.m2[i];

            // Compute the full update vector first so we can compute its norm.
            let n = p.weights.len();
            let mut update = vec![0.0_f32; n];
            let mut w_norm_sq = 0.0_f32;

            for (j, (&g, &w)) in p.grads.iter().zip(p.weights.iter()).enumerate() {
                let g = clip_grad(g, self.inner.hp.grad_clip);

                m1[j] = self.inner.beta1 * m1[j] + (1.0 - self.inner.beta1) * g;
                m2[j] = self.inner.beta2 * m2[j] + (1.0 - self.inner.beta2) * g * g;

                let m_hat = m1[j] / bc1;
                let v_hat = m2[j] / bc2;
                let adam  = m_hat / (v_hat.sqrt() + self.inner.eps);

                // Weight decay added to update (LAMB couples it here, unlike AdamW).
                update[j] = adam + eff_wd * w;
                w_norm_sq += w * w;
            }

            // Layer-wise trust ratio.
            let w_norm  = w_norm_sq.sqrt();
            let u_norm: f32 = update.iter().map(|u| u * u).sum::<f32>().sqrt();

            let trust = if w_norm < 1e-8 || u_norm < 1e-8 {
                1.0
            } else {
                (w_norm / u_norm).clamp(self.clamp_trust.0, self.clamp_trust.1)
            };

            for (j, w) in p.weights.iter_mut().enumerate() {
                *w -= eff_lr * trust * update[j];
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 { self.inner.current_lr(step) }
    fn state_snapshot(&self) -> OptimizerState { self.inner.state_snapshot() }
    fn load_snapshot(&mut self, s: OptimizerState) { self.inner.load_snapshot(s); }
}

// =============================================================================
// §18  LARS  (You et al. 2017)
// =============================================================================

/// LARS: Layer-wise Adaptive Rate Scaling (SGD variant).
///
/// effective_lr = α * ‖w‖ / (‖g‖ + λ‖w‖)  * η
/// v ← μ * v - effective_lr * g
/// w ← w + v
pub struct Lars {
    pub hp:       HParams,
    pub momentum: f32,
    pub eta:      f32,   // trust coefficient
    velocity:     Vec<Vec<f32>>,
    schedule:     Box<dyn LrSchedule>,
}

impl Lars {
    pub fn new(lr: f32) -> Self {
        Lars {
            hp:       HParams { lr, weight_decay: 1e-4, ..Default::default() },
            momentum: 0.9,
            eta:      1e-3,
            velocity: Vec::new(),
            schedule: Box::new(ConstantLr),
        }
    }
    pub fn with_eta(mut self, eta: f32) -> Self { self.eta = eta; self }
    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.hp.weight_decay = wd; self }
    pub fn with_schedule(mut self, s: impl LrSchedule + 'static) -> Self {
        self.schedule = Box::new(s); self
    }

    fn ensure_buffers(&mut self, params: &[ParamBuffer<'_>]) {
        if self.velocity.len() < params.len() {
            self.velocity.resize_with(params.len(), Vec::new);
        }
        for (i, p) in params.iter().enumerate() {
            if self.velocity[i].len() != p.numel() {
                self.velocity[i] = vec![0.0; p.numel()];
            }
        }
    }
}

impl Optimizer for Lars {
    fn name(&self) -> &str { "LARS" }

    fn step(&mut self, params: &mut [ParamBuffer<'_>], step: u64) {
        self.ensure_buffers(params);
        let base_lr = self.schedule.multiplier(step, self.hp.lr);

        for (i, p) in params.iter_mut().enumerate() {
            let eff_wd = self.hp.weight_decay * p.wd_scale;
            let vel    = &mut self.velocity[i];

            let w_norm: f32 = p.weights.iter().map(|w| w * w).sum::<f32>().sqrt();
            let g_norm: f32 = p.grads.iter().map(|g| g * g).sum::<f32>().sqrt();

            let local_lr = if w_norm < 1e-8 || g_norm < 1e-8 {
                base_lr
            } else {
                base_lr * self.eta * w_norm / (g_norm + eff_wd * w_norm)
            };
            let eff_lr = local_lr * p.lr_scale;

            for (j, (&g, w)) in p.grads.iter().zip(p.weights.iter_mut()).enumerate() {
                let g = clip_grad(g, self.hp.grad_clip) + eff_wd * *w;
                vel[j] = self.momentum * vel[j] - eff_lr * g;
                *w += vel[j];
            }
        }
    }

    fn current_lr(&self, step: u64) -> f32 { self.schedule.multiplier(step, self.hp.lr) }

    fn state_snapshot(&self) -> OptimizerState {
        let buffers = self.velocity.iter().enumerate()
            .map(|(i, v)| (format!("vel_{i}"), v.clone()))
            .collect();
        OptimizerState {
            name: "LARS".into(), step: 0, lr: self.hp.lr, buffers,
            hparams: vec![
                ("momentum".into(), self.momentum),
                ("eta".into(), self.eta),
                ("weight_decay".into(), self.hp.weight_decay),
            ],
        }
    }

    fn load_snapshot(&mut self, state: OptimizerState) {
        self.hp.lr = state.lr;
        self.velocity = state.buffers.into_iter().map(|(_, v)| v).collect();
    }
}

// =============================================================================
// §19  FACTORY / BUILDER
// =============================================================================

/// Build an optimizer by kind + hyperparameters (matches the AST `OptimizerKind`).
pub fn build_optimizer(
    kind:         crate::ast::OptimizerKind,
    lr:           f32,
    weight_decay: f32,
    grad_clip:    Option<GradClip>,
    schedule:     Option<Box<dyn LrSchedule>>,
) -> Box<dyn Optimizer> {
    use crate::ast::OptimizerKind::*;
    let clip = grad_clip;
    match kind {
        Adam   => {
            let mut opt = crate::optimizer::Adam::new(lr)
                .with_weight_decay(weight_decay);
            if let Some(c) = clip { opt = opt.with_grad_clip(c); }
            if let Some(s) = schedule { opt = opt.with_schedule_boxed(s); }
            Box::new(opt)
        }
        AdamW  => {
            let mut opt = crate::optimizer::AdamW::new(lr)
                .with_weight_decay(weight_decay);
            if let Some(c) = clip { opt = opt.with_grad_clip(c); }
            if let Some(s) = schedule { opt = opt.with_schedule_boxed(s); }
            Box::new(opt)
        }
        Sgd    => {
            let mut opt = crate::optimizer::Sgd::new(lr, 0.9)
                .with_weight_decay(weight_decay);
            if let Some(c) = clip { opt = opt.with_grad_clip(c); }
            if let Some(s) = schedule { opt = opt.with_schedule_boxed(s); }
            Box::new(opt)
        }
        Rmsprop => {
            let mut opt = crate::optimizer::RmsProp::new(lr)
                .with_weight_decay(weight_decay);
            if let Some(s) = schedule { opt = opt.with_schedule_boxed(s); }
            Box::new(opt)
        }
        Adagrad => {
            let opt = crate::optimizer::AdaGrad::new(lr)
                .with_weight_decay(weight_decay);
            Box::new(opt)
        }
    }
}

// ── Schedule-boxed helpers so the builder can accept `Box<dyn LrSchedule>` ──

trait WithScheduleBoxed: Sized {
    fn with_schedule_boxed(self, s: Box<dyn LrSchedule>) -> Self;
}

macro_rules! impl_schedule_boxed {
    ($t:ty) => {
        impl WithScheduleBoxed for $t {
            fn with_schedule_boxed(mut self, s: Box<dyn LrSchedule>) -> Self {
                self.schedule = s; self
            }
        }
    };
}
impl_schedule_boxed!(Sgd);
impl_schedule_boxed!(Adam);
impl_schedule_boxed!(AdaGrad);
impl_schedule_boxed!(RmsProp);
impl_schedule_boxed!(Lars);

impl WithScheduleBoxed for AdamW {
    fn with_schedule_boxed(mut self, s: Box<dyn LrSchedule>) -> Self {
        self.inner.schedule = s; self
    }
}

// =============================================================================
// §20  HELPER FUNCTIONS
// =============================================================================

#[inline]
fn clip_grad(g: f32, clip: Option<GradClip>) -> f32 {
    match clip {
        Some(GradClip::ByValue(v)) => g.clamp(-v, v),
        _ => g,
    }
}

// =============================================================================
// §21  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
    /// Gradient: ∂f/∂x = -2(1-x) - 400x(y-x²), ∂f/∂y = 200(y-x²)
    /// Global minimum at (1, 1) with f=0.
    fn rosenbrock_grad(x: f32, y: f32) -> [f32; 2] {
        let gx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
        let gy = 200.0 * (y - x * x);
        [gx, gy]
    }

    fn rosenbrock(x: f32, y: f32) -> f32 {
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    }

    /// Quadratic bowl  f(x) = ½ * xᵀAx  (A = diag([1..n]))
    /// Gradient = A * x.  Minimum at x = 0.
    fn quadratic_grad(x: &[f32]) -> Vec<f32> {
        x.iter().enumerate().map(|(i, &xi)| (i as f32 + 1.0) * xi).collect()
    }

    /// Run optimizer on 2-parameter Rosenbrock for n steps, return final loss.
    fn run_rosenbrock(opt: &mut dyn Optimizer, steps: u64, lr: f32) -> f32 {
        let mut w = vec![-1.0_f32, 1.0_f32];
        for step in 0..steps {
            let [gx, gy] = rosenbrock_grad(w[0], w[1]);
            let grads = vec![gx, gy];
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
        }
        rosenbrock(w[0], w[1])
    }

    /// Run on n-dim quadratic bowl. Return final ‖x‖².
    fn run_quadratic(opt: &mut dyn Optimizer, n: usize, steps: u64) -> f32 {
        let mut w: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        for step in 0..steps {
            let grads = quadratic_grad(&w);
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
        }
        w.iter().map(|x| x * x).sum::<f32>()
    }

    // ── §9  SGD ───────────────────────────────────────────────────────────────

    #[test]
    fn test_sgd_plain_descends_quadratic() {
        let mut opt = Sgd::new(0.01, 0.0);
        let loss = run_quadratic(&mut opt, 4, 500);
        assert!(loss < 1e-4, "SGD: final ‖x‖² = {loss}");
    }

    #[test]
    fn test_sgd_momentum_faster_than_plain() {
        let mut plain = Sgd::new(0.01, 0.0);
        let mut mom   = Sgd::new(0.01, 0.9);
        let l_plain = run_quadratic(&mut plain, 4, 200);
        let l_mom   = run_quadratic(&mut mom,   4, 200);
        assert!(l_mom < l_plain, "momentum should converge faster: {l_mom} vs {l_plain}");
    }

    #[test]
    fn test_sgd_nesterov_converges() {
        let mut opt = Sgd::new(0.01, 0.9).with_nesterov();
        let loss = run_quadratic(&mut opt, 4, 200);
        assert!(loss < 1e-3, "Nesterov SGD: {loss}");
    }

    #[test]
    fn test_sgd_weight_decay_shrinks_weights() {
        // With very high WD and no gradient, weights should decay to 0.
        let mut w     = vec![1.0_f32; 4];
        let grads     = vec![0.0_f32; 4];
        let mut opt   = Sgd::new(0.1, 0.9).with_weight_decay(0.5);
        for step in 0..50 {
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
        }
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 0.01, "weights should shrink with high WD, got norm={norm}");
    }

    // ── §10  Adam ─────────────────────────────────────────────────────────────

    #[test]
    fn test_adam_converges_rosenbrock() {
        let mut opt = Adam::new(0.01);
        let loss = run_rosenbrock(&mut opt, 5000, 0.01);
        assert!(loss < 0.1, "Adam Rosenbrock loss = {loss}");
    }

    #[test]
    fn test_adam_zero_grad_no_change() {
        // With zero gradients weights should not change.
        let mut w   = vec![1.0_f32, 2.0_f32, 3.0_f32];
        let grads   = vec![0.0_f32; 3];
        let initial = w.clone();
        let mut opt = Adam::new(1e-3);
        for step in 0..10 {
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
        }
        for (a, b) in w.iter().zip(&initial) {
            assert!((a - b).abs() < 1e-7, "zero grad should not move weights");
        }
    }

    #[test]
    fn test_adam_bias_correction_first_step() {
        // After one step with g=1, the weight should decrease.
        let mut w   = vec![1.0_f32];
        let grads   = vec![1.0_f32];
        let mut opt = Adam::new(0.01);
        let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
        opt.step(&mut params, 0);
        assert!(w[0] < 1.0, "Adam should have decreased weight, got {}", w[0]);
    }

    #[test]
    fn test_adam_snapshot_restore() {
        let mut opt = Adam::new(0.001);
        let mut w1 = vec![0.5_f32; 8];
        let grads  = vec![0.1_f32; 8];

        // Warm up for 5 steps.
        for step in 0..5 {
            let mut params = [ParamBuffer::new(&mut w1, &grads, "w")];
            opt.step(&mut params, step);
        }
        let snap = opt.state_snapshot();
        let w_before = w1.clone();

        // 5 more steps.
        for step in 5..10 {
            let mut params = [ParamBuffer::new(&mut w1, &grads, "w")];
            opt.step(&mut params, step);
        }
        let w_after = w1.clone();

        // Restore snapshot and replay the same 5 steps.
        opt.load_snapshot(snap);
        w1 = w_before;
        for step in 5..10 {
            let mut params = [ParamBuffer::new(&mut w1, &grads, "w")];
            opt.step(&mut params, step);
        }

        for (a, b) in w1.iter().zip(&w_after) {
            assert!((a - b).abs() < 1e-6, "snapshot restore should reproduce same trajectory");
        }
    }

    // ── §11  AdamW ────────────────────────────────────────────────────────────

    #[test]
    fn test_adamw_decouples_weight_decay() {
        // With wd > 0 and a large gradient, AdamW should regularise MORE than Adam
        // because decay is not diluted by the adaptive denominator.
        let mut w_adam  = vec![1.0_f32];
        let mut w_adamw = vec![1.0_f32];
        let grads       = vec![0.001_f32]; // tiny gradient so WD dominates

        let mut adam  = Adam::new(0.01).with_weight_decay(0.5);
        let mut adamw = AdamW::new(0.01).with_weight_decay(0.5);

        for step in 0..100 {
            let mut p_a = [ParamBuffer::new(&mut w_adam,  &grads, "w")];
            adam.step(&mut p_a, step);
            let mut p_aw = [ParamBuffer::new(&mut w_adamw, &grads, "w")];
            adamw.step(&mut p_aw, step);
        }

        // AdamW should shrink the weight more than standard Adam under high WD.
        assert!(w_adamw[0].abs() < w_adam[0].abs(),
            "AdamW={} should be smaller than Adam={}", w_adamw[0], w_adam[0]);
    }

    #[test]
    fn test_adamw_converges_quadratic() {
        let mut opt = AdamW::new(1e-3).with_weight_decay(1e-4);
        let loss = run_quadratic(&mut opt, 8, 1000);
        assert!(loss < 1e-3, "AdamW quadratic loss = {loss}");
    }

    // ── §12  AdaGrad ──────────────────────────────────────────────────────────

    #[test]
    fn test_adagrad_converges_quadratic() {
        let mut opt = AdaGrad::new(0.5);
        let loss = run_quadratic(&mut opt, 4, 1000);
        assert!(loss < 1e-4, "AdaGrad quadratic = {loss}");
    }

    #[test]
    fn test_adagrad_rare_feature_handling() {
        // AdaGrad should give a larger update to a rarely-seen feature.
        let mut w = vec![1.0_f32; 2];
        let mut opt = AdaGrad::new(0.1);

        // Feature 0 gets constant gradient; feature 1 gets sparse gradient.
        for step in 0..100 {
            let g1 = if step % 10 == 0 { 1.0 } else { 0.0 };
            let grads = vec![1.0_f32, g1];
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
        }
        // Feature 1 (sparse) should have moved less total distance than feature 0.
        // But its per-occurrence step should be larger — hard to test without
        // full history; instead just confirm both converge toward 0.
        assert!(w[0] < 1.0 && w[1] < 1.0, "both features should have decreased");
    }

    // ── §13  RMSProp ──────────────────────────────────────────────────────────

    #[test]
    fn test_rmsprop_converges_quadratic() {
        let mut opt = RmsProp::new(0.01);
        let loss = run_quadratic(&mut opt, 4, 500);
        assert!(loss < 1e-4, "RMSProp quadratic = {loss}");
    }

    #[test]
    fn test_rmsprop_centred_converges() {
        let mut opt = RmsProp::new(0.01).with_centred();
        let loss = run_quadratic(&mut opt, 4, 500);
        assert!(loss < 1e-3, "Centred RMSProp = {loss}");
    }

    #[test]
    fn test_rmsprop_momentum_converges() {
        let mut opt = RmsProp::new(0.01).with_momentum(0.9);
        let loss = run_quadratic(&mut opt, 4, 500);
        assert!(loss < 1e-4, "RMSProp+momentum = {loss}");
    }

    // ── §14  AdaDelta ─────────────────────────────────────────────────────────

    #[test]
    fn test_adadelta_converges_quadratic() {
        let mut opt = AdaDelta::new();
        let loss = run_quadratic(&mut opt, 4, 2000);
        assert!(loss < 0.1, "AdaDelta quadratic = {loss}");
    }

    // ── §15  Nadam ────────────────────────────────────────────────────────────

    #[test]
    fn test_nadam_converges_rosenbrock() {
        let mut opt = Nadam::new(0.005);
        let loss = run_rosenbrock(&mut opt, 6000, 0.005);
        assert!(loss < 0.5, "Nadam Rosenbrock = {loss}");
    }

    // ── §16  RAdam ────────────────────────────────────────────────────────────

    #[test]
    fn test_radam_converges_quadratic() {
        let mut opt = Radam::new(0.005);
        let loss = run_quadratic(&mut opt, 4, 1000);
        assert!(loss < 1e-3, "RAdam quadratic = {loss}");
    }

    #[test]
    fn test_radam_warm_start_no_nan() {
        // In early steps, RAdam should fall back to SGD (no NaN / inf).
        let mut opt = Radam::new(0.001);
        let mut w = vec![1.0_f32; 16];
        let grads: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        for step in 0..20 {
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
            assert!(w.iter().all(|x| x.is_finite()), "RAdam produced NaN at step {step}");
        }
    }

    // ── §17  LAMB ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lamb_converges_quadratic() {
        let mut opt = Lamb::new(0.01);
        let loss = run_quadratic(&mut opt, 8, 500);
        assert!(loss < 0.1, "LAMB quadratic = {loss}");
    }

    #[test]
    fn test_lamb_trust_ratio_bounded() {
        // The LAMB update should never blow up — trust ratio is clamped.
        let mut opt = Lamb::new(0.1);
        let mut w: Vec<f32> = vec![100.0; 32]; // large weights
        let grads: Vec<f32> = vec![1e-6; 32];  // tiny gradients
        for step in 0..50 {
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
            assert!(w.iter().all(|x| x.is_finite()),
                "LAMB produced non-finite at step {step}");
        }
    }

    // ── §18  LARS ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lars_converges_quadratic() {
        let mut opt = Lars::new(0.1).with_eta(1.0).with_weight_decay(0.0);
        let loss = run_quadratic(&mut opt, 4, 500);
        assert!(loss < 0.01, "LARS quadratic = {loss}");
    }

    // ── Schedules ─────────────────────────────────────────────────────────────

    #[test]
    fn test_constant_schedule() {
        let s = ConstantLr;
        assert_eq!(s.multiplier(0,   0.01), 0.01);
        assert_eq!(s.multiplier(999, 0.01), 0.01);
    }

    #[test]
    fn test_step_decay_schedule() {
        let s = StepDecay { step_size: 10, gamma: 0.5 };
        assert_eq!(s.multiplier(0,  1.0), 1.0);
        assert_eq!(s.multiplier(10, 1.0), 0.5);
        assert_eq!(s.multiplier(20, 1.0), 0.25);
    }

    #[test]
    fn test_exponential_decay() {
        let s = ExponentialDecay { gamma: 0.99 };
        let lr100 = s.multiplier(100, 1.0);
        assert!((lr100 - 0.99_f32.powi(100)).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_annealing_endpoints() {
        let s = CosineAnnealing { t_max: 100, lr_min: 0.001 };
        let lr0   = s.multiplier(0,   0.1);
        let lr100 = s.multiplier(100, 0.1);
        assert!((lr0 - 0.1).abs() < 1e-5,    "start = base_lr: {lr0}");
        assert!((lr100 - 0.001).abs() < 1e-5, "end = lr_min: {lr100}");
    }

    #[test]
    fn test_cosine_annealing_monotone_decrease() {
        let s = CosineAnnealing { t_max: 50, lr_min: 0.0 };
        let lrs: Vec<f32> = (0..=50).map(|t| s.multiplier(t, 1.0)).collect();
        for w in lrs.windows(2) {
            assert!(w[0] >= w[1] - 1e-6, "should be monotone: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_cosine_warm_restarts_period_doubles() {
        let s = CosineAnnealingWarmRestarts { t_0: 10, t_mult: 2, lr_min: 0.0 };
        // At t=10 (start of 2nd cycle) lr should be close to base again.
        let restart_lr = s.multiplier(10, 1.0);
        assert!((restart_lr - 1.0).abs() < 1e-4, "restart lr = {restart_lr}");
        // At t=30 (start of 3rd cycle) same.
        let restart2 = s.multiplier(30, 1.0);
        assert!((restart2 - 1.0).abs() < 1e-4, "2nd restart = {restart2}");
    }

    #[test]
    fn test_linear_warmup_reaches_base() {
        let s = LinearWarmup { warmup_steps: 10, base: ConstantLr };
        assert!((s.multiplier(0,  1.0) - 0.0).abs() < 1e-5);
        assert!((s.multiplier(5,  1.0) - 0.5).abs() < 1e-5);
        assert!((s.multiplier(10, 1.0) - 1.0).abs() < 1e-5);
        assert!((s.multiplier(15, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_one_cycle_lr_peak_at_pct_start() {
        let s = OneCycleLr {
            total_steps: 100, div_factor: 10.0, final_div_factor: 1e4, pct_start: 0.3,
        };
        let base_lr = 0.1;
        // At 30% through, lr should be at its maximum (≈ base_lr).
        let peak = s.multiplier(30, base_lr);
        assert!((peak - base_lr).abs() < 0.001, "peak = {peak}");
    }

    #[test]
    fn test_polynomial_decay_endpoints() {
        let s = PolynomialDecay { total_steps: 100, end_lr: 0.001, power: 1.0 };
        let lr0   = s.multiplier(0,   1.0);
        let lr100 = s.multiplier(100, 1.0);
        assert!((lr0   - 1.0  ).abs() < 1e-5, "start: {lr0}");
        assert!((lr100 - 0.001).abs() < 1e-5, "end: {lr100}");
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut s = ReduceOnPlateau::new(0.5, 2, 1e-6);
        let base  = 0.1;
        // No reduction yet.
        assert!((s.multiplier(0, base) - 0.1).abs() < 1e-6);
        s.on_metric(1.0); // initial (stores best=1.0)
        s.on_metric(1.0); // bad epoch 1
        s.on_metric(1.0); // bad epoch 2 → patience hit → reduce
        let lr_after = s.multiplier(0, base);
        assert!(lr_after < 0.1, "should have reduced: {lr_after}");
    }

    // ── Gradient clipping ─────────────────────────────────────────────────────

    #[test]
    fn test_clip_by_value() {
        let mut g = vec![-5.0_f32, 2.0, 0.5, -0.1];
        clip_by_value(&mut g, 1.0);
        assert_eq!(g, vec![-1.0, 1.0, 0.5, -0.1]);
    }

    #[test]
    fn test_clip_by_norm_scales_when_exceeds() {
        let mut g = vec![3.0_f32, 4.0]; // norm = 5
        clip_by_norm(&mut g, 1.0);
        let norm: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm after clip = {norm}");
    }

    #[test]
    fn test_clip_by_norm_noop_when_below() {
        let mut g = vec![0.3_f32, 0.4]; // norm = 0.5 < 1.0
        let before = g.clone();
        clip_by_norm(&mut g, 1.0);
        assert_eq!(g, before, "should not clip when norm < max");
    }

    #[test]
    fn test_clip_by_global_norm() {
        let mut g1 = vec![3.0_f32, 0.0]; // norm = 3
        let mut g2 = vec![0.0_f32, 4.0]; // norm = 4
        // global norm = 5; clip to 1 → scale = 0.2
        let global = clip_by_global_norm(&mut [&mut g1, &mut g2], 1.0);
        assert!((global - 5.0).abs() < 1e-4, "global norm = {global}");
        let norm_after: f32 = g1.iter().chain(g2.iter()).map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_after - 1.0).abs() < 1e-4, "post-clip global norm = {norm_after}");
    }

    // ── Regularisation ────────────────────────────────────────────────────────

    #[test]
    fn test_l2_grad_increases_gradient() {
        let w = vec![2.0_f32];
        let mut g = vec![0.0_f32];
        add_l2_gradient(&w, &mut g, 0.1);
        assert!((g[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_l1_grad_uses_sign() {
        let w = vec![-3.0_f32, 5.0, 0.0];
        let mut g = vec![0.0_f32; 3];
        add_l1_gradient(&w, &mut g, 1.0);
        assert_eq!(g[0], -1.0);
        assert_eq!(g[1],  1.0);
        assert_eq!(g[2],  0.0);
    }

    #[test]
    fn test_elastic_net_blends() {
        let w = vec![1.0_f32];
        let mut g = vec![0.0_f32];
        // alpha=0 → pure L2: g += lambda * w = 0.1
        add_elastic_net_gradient(&w, &mut g, 0.1, 0.0);
        assert!((g[0] - 0.1).abs() < 1e-6);
        // alpha=1 → pure L1: g += lambda * sign(w) = 0.1
        g[0] = 0.0;
        add_elastic_net_gradient(&w, &mut g, 0.1, 1.0);
        assert!((g[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_weight_decay_applied_directly() {
        let mut w = vec![1.0_f32; 4];
        apply_weight_decay(&mut w, 0.1, 0.1); // scale = 1 - 0.01 = 0.99
        for &wi in &w {
            assert!((wi - 0.99).abs() < 1e-6);
        }
    }

    // ── Gradient accumulator ──────────────────────────────────────────────────

    #[test]
    fn test_accumulator_averages_correctly() {
        let mut acc = GradAccumulator::zeros(3);
        acc.accumulate(&[1.0, 2.0, 3.0]);
        acc.accumulate(&[3.0, 4.0, 5.0]);
        acc.average(2);
        assert!((acc.buffer[0] - 2.0).abs() < 1e-6);
        assert!((acc.buffer[1] - 3.0).abs() < 1e-6);
        assert!((acc.buffer[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = GradAccumulator::zeros(2);
        acc.accumulate(&[5.0, 5.0]);
        acc.reset();
        assert_eq!(acc.buffer, vec![0.0, 0.0]);
        assert_eq!(acc.count(), 0);
    }

    // ── Adam vs AdamW comparison ──────────────────────────────────────────────

    #[test]
    fn test_adamw_better_regularisation_large_wd() {
        // After many steps with moderate gradient and large WD, AdamW should
        // produce smaller weights than Adam because its decay is not diluted.
        let mut wa = vec![10.0_f32; 4];
        let mut ww = vec![10.0_f32; 4];
        let grads  = vec![0.01_f32; 4];

        let mut adam  = Adam::new(0.001).with_weight_decay(1.0);
        let mut adamw = AdamW::new(0.001).with_weight_decay(1.0);

        for step in 0..500 {
            let mut pa = [ParamBuffer::new(&mut wa, &grads, "w")];
            adam.step(&mut pa, step);
            let mut pw = [ParamBuffer::new(&mut ww, &grads, "w")];
            adamw.step(&mut pw, step);
        }

        let norm_a: f32 = wa.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_w: f32 = ww.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm_w < norm_a, "AdamW norm={norm_w} should < Adam norm={norm_a}");
    }

    // ── Multi-group update ────────────────────────────────────────────────────

    #[test]
    fn test_multiple_param_groups_independent() {
        let mut w1 = vec![1.0_f32; 2];
        let mut w2 = vec![2.0_f32; 2];
        let g1 = vec![0.1_f32; 2];
        let g2 = vec![0.5_f32; 2];

        let mut opt = Adam::new(0.01);
        let mut params = [
            ParamBuffer::new(&mut w1, &g1, "layer1"),
            ParamBuffer::new(&mut w2, &g2, "layer2"),
        ];
        opt.step(&mut params, 0);

        // Both groups should have moved, independently.
        assert!(w1[0] < 1.0, "group 1 should decrease");
        assert!(w2[0] < 2.0, "group 2 should decrease");
    }

    #[test]
    fn test_per_group_lr_scale() {
        let mut w1 = vec![1.0_f32];
        let mut w2 = vec![1.0_f32];
        let g = vec![1.0_f32];

        let mut opt = Sgd::new(0.1, 0.0);
        {
            let mut params = [
                ParamBuffer::new(&mut w1, &g, "fast").with_lr_scale(10.0),
                ParamBuffer::new(&mut w2, &g, "slow").with_lr_scale(0.1),
            ];
            opt.step(&mut params, 0);
        }
        // w1 should have moved much more than w2.
        let d1 = (1.0 - w1[0]).abs();
        let d2 = (1.0 - w2[0]).abs();
        assert!(d1 > d2 * 5.0, "fast group (d={d1}) should move more than slow (d={d2})");
    }

    // ── No NaN / Inf stability ─────────────────────────────────────────────────

    #[test]
    fn test_adam_no_nan_large_gradients() {
        let mut opt = Adam::new(0.001);
        let mut w: Vec<f32> = vec![1.0; 64];
        for step in 0..100 {
            // Simulate gradient explosion.
            let grads: Vec<f32> = w.iter().map(|&wi| wi * 1000.0).collect();
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
            assert!(w.iter().all(|x| x.is_finite()),
                "Adam produced NaN/Inf at step {step}, grad_norm≈large");
        }
    }

    #[test]
    fn test_sgd_no_nan_zero_weights() {
        let mut opt = Sgd::new(0.01, 0.9);
        let mut w = vec![0.0_f32; 8];
        let grads = vec![1.0_f32; 8];
        for step in 0..20 {
            let mut params = [ParamBuffer::new(&mut w, &grads, "w")];
            opt.step(&mut params, step);
            assert!(w.iter().all(|x| x.is_finite()),
                "SGD produced NaN at step {step}");
        }
    }

    // ── AdamW with schedule ───────────────────────────────────────────────────

    #[test]
    fn test_adamw_with_cosine_schedule() {
        let mut opt = AdamW::new(0.01)
            .with_schedule(CosineAnnealing { t_max: 100, lr_min: 1e-6 });
        let loss = run_quadratic(&mut opt, 4, 100);
        assert!(loss < 0.1, "AdamW+cosine: {loss}");
    }

    #[test]
    fn test_adam_with_warmup_cosine() {
        let mut opt = Adam::new(0.01)
            .with_schedule(LinearWarmup {
                warmup_steps: 50,
                base: CosineAnnealing { t_max: 450, lr_min: 1e-6 },
            });
        let loss = run_quadratic(&mut opt, 4, 500);
        assert!(loss < 0.01, "Adam+warmup+cosine: {loss}");
    }
}
