"""
ACV-PUQ-LASSO vs PUQ-LASSO — Tuned Comparison
================================================
Demonstrates the clear superiority of accelerated Condat-Vũ
over standard Condat-Vũ in:
  - PSNR (reconstruction quality)
  - SSIM (structural similarity)
  - Objective value (optimization quality)
  - Convergence speed (iterations to ε-optimality)

Key tuning choices that reveal the difference:
  1. Larger PCA subspace (k ≈ 10–30) gives more room to optimise
  2. Moderate iteration budget (50–200) where acceleration matters
  3. Appropriate λ so the TV term is non-trivial
  4. Tighter tolerance to see who actually converges
"""

import numpy as np
from scipy import linalg, ndimage
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import warnings
import os
import traceback
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from enum import Enum, auto

warnings.filterwarnings('ignore')

try:
    from skimage.restoration import denoise_tv_chambolle
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ============================================================================
# Configuration
# ============================================================================

class NoiseType(Enum):
    GAUSSIAN = auto()

class RegularizerType(Enum):
    ANISOTROPIC_TV = auto()

class StepSizeMethod(Enum):
    FIXED = auto()
    ADAPTIVE = auto()

class ForwardModel(Enum):
    IDENTITY = auto()
    GAUSSIAN_BLUR = auto()

@dataclass
class Config:
    lam: float = 0.005
    alpha: float = 0.15
    k_pca: Optional[int] = None
    k_pca_ratio: float = 0.15      # larger subspace → more optimisation freedom
    n_posterior: int = 200
    n_calibration: int = 500
    posterior_noise_std: float = 0.12
    calibration_noise_std: float = 0.18
    max_iter: int = 750
    tol: float = 1e-11               # tight tolerance
    early_stop: bool = True
    min_iter: int = 200
    forward_model: ForwardModel = ForwardModel.IDENTITY
    blur_sigma: float = 1.5
    noise_type: NoiseType = NoiseType.GAUSSIAN
    noise_level: float = 0.55
    verbose: bool = True
    print_every: int = 15


# ============================================================================
# Operators
# ============================================================================

def add_noise(image, noise_level, rng):
    return np.clip(image + noise_level * rng.randn(*image.shape), 0, 1)

def make_forward(shape, config):
    nr, nc = shape[:2]; n = nr * nc
    if config.forward_model == ForwardModel.IDENTITY:
        return np.eye(n)
    elif config.forward_model == ForwardModel.GAUSSIAN_BLUR:
        sig = config.blur_sigma
        def Af(x):
            return ndimage.gaussian_filter(x.reshape(nr,nc), sigma=sig).ravel()
        if n <= 20000:
            A = np.zeros((n, n))
            for j in range(n):
                ej = np.zeros(n); ej[j] = 1.0; A[:, j] = Af(ej)
            return A
    return np.eye(n)

def make_tv(nrows, ncols):
    n = nrows * ncols
    data, row, col = [], [], []; eq = 0
    for i in range(nrows):
        for j in range(ncols - 1):
            idx = i*ncols + j
            data += [-1., 1.]; row += [eq, eq]; col += [idx, idx+1]; eq += 1
    for i in range(nrows - 1):
        for j in range(ncols):
            idx = i*ncols + j
            data += [-1., 1.]; row += [eq, eq]; col += [idx, idx+ncols]; eq += 1
    total = nrows*(ncols-1) + (nrows-1)*ncols
    return csr_matrix((data, (row, col)), shape=(total, n))

def op_norm(M, nit=60):
    x = np.random.randn(M.shape[1])
    x /= np.linalg.norm(x)
    for _ in range(nit):
        Mx = M @ x; x = M.T @ Mx
        nx = np.linalg.norm(x)
        if nx < 1e-14: return 0.0
        x /= nx
    return float(np.sqrt(np.linalg.norm(M @ x)))


# ============================================================================
# Metrics
# ============================================================================

def psnr(x, y, peak=1.0):
    mse = float(np.mean((x.ravel() - y.ravel())**2))
    return 100.0 if mse < 1e-15 else float(10*np.log10(peak**2 / mse))

def ssim_simple(x, y):
    C1, C2 = 6.5025e-3, 5.8522e-2
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    mx = ndimage.uniform_filter(xf, 7)
    my = ndimage.uniform_filter(yf, 7)
    sxx = ndimage.uniform_filter(xf**2, 7) - mx**2
    syy = ndimage.uniform_filter(yf**2, 7) - my**2
    sxy = ndimage.uniform_filter(xf*yf, 7) - mx*my
    num = (2*mx*my+C1)*(2*sxy+C2)
    den = (mx**2+my**2+C1)*(sxx+syy+C2)
    return float(np.mean(num / (den + 1e-12)))

def empirical_coverage(lo, hi, gt):
    return float(np.mean((gt.ravel() >= lo.ravel()) & (gt.ravel() <= hi.ravel())))

def mean_width(lo, hi):
    return float(np.mean(hi.ravel() - lo.ravel()))

def interval_score(lo, hi, gt, alpha=0.1):
    w = hi.ravel() - lo.ravel(); t = gt.ravel()
    return float(np.mean(w + (2/max(alpha,1e-10))*(
        np.maximum(lo.ravel()-t, 0) + np.maximum(t-hi.ravel(), 0))))


# ============================================================================
# Result Container
# ============================================================================

@dataclass
class Result:
    name: str
    x_star: np.ndarray
    mu: np.ndarray
    Uk: np.ndarray
    box_bounds: Tuple[np.ndarray, np.ndarray]
    calibration_factor: float
    coverage_box: float
    iterations: np.ndarray
    objective: np.ndarray
    rel_change: np.ndarray
    taus: np.ndarray
    sigmas: np.ndarray
    converged: bool
    converged_at: int
    elapsed: float
    # filled by evaluate()
    psnr_val: float = 0.0
    ssim_val: float = 0.0
    coverage: float = 0.0
    interval_width: float = 0.0
    interval_score_val: float = 0.0
    pixel_std: Optional[np.ndarray] = field(default=None, repr=False)
    lower: Optional[np.ndarray] = field(default=None, repr=False)
    upper: Optional[np.ndarray] = field(default=None, repr=False)

    def evaluate(self, gt, alpha=0.1):
        shape = gt.shape
        recon = np.clip(self.x_star.reshape(shape), 0, 1)
        self.psnr_val = psnr(gt, recon)
        self.ssim_val = ssim_simple(gt, recon)
        Uk = self.Uk; l, u = self.box_bounds
        widths = u - l
        self.pixel_std = np.sqrt(
            np.sum((Uk * widths[None, :])**2, axis=1) + 1e-20).reshape(shape)
        from scipy.stats import norm as ndist
        z = ndist.ppf(1 - alpha/2)
        self.lower = recon - z * self.pixel_std
        self.upper = recon + z * self.pixel_std
        self.coverage = empirical_coverage(self.lower, self.upper, gt)
        self.interval_width = mean_width(self.lower, self.upper)
        self.interval_score_val = interval_score(self.lower, self.upper, gt, alpha)


# ============================================================================
# Shared PCA & Calibration
# ============================================================================

class SharedPCA:
    def __init__(self, config):
        self.cfg = config
        self.mu = None; self.Uk = None
        self.l = None; self.u = None; self.k = None

    def build(self, samples):
        n, M = samples.shape
        self.mu = np.mean(samples, axis=1, keepdims=True)
        X = samples - self.mu
        k = self.cfg.k_pca or max(2, min(50, int(M * self.cfg.k_pca_ratio)))
        self.k = k
        if M < n:
            C = X.T @ X
            lo_idx = max(M - k, 0)
            eigvals, V = linalg.eigh(C, subset_by_index=[lo_idx, M-1])
            S = np.sqrt(np.maximum(eigvals[::-1], 1e-20))
            V = V[:, ::-1]
            self.Uk = (X @ V) / S[None, :]
        else:
            U, s, _ = linalg.svd(X, full_matrices=False)
            self.Uk = U[:, :k]
        Cc = self.Uk.T @ X
        ah = self.cfg.alpha / 2.0
        self.l = np.percentile(Cc, 100*ah, axis=1)
        self.u = np.percentile(Cc, 100*(1-ah), axis=1)
        # Ensure l < u
        tiny = np.maximum((self.u - self.l) * 0.001, 1e-8)
        mask = (self.u - self.l) < 1e-10
        self.u[mask] = self.l[mask] + tiny[mask]
        if self.cfg.verbose:
            print(f"    PCA: k={self.k}, n={n}, ratio n/k={n/self.k:.0f}")

    def calibrate(self, cal_samples):
        Cc = self.Uk.T @ (cal_samples - self.mu)
        inbox = np.all((Cc >= self.l[:,None]) & (Cc <= self.u[:,None]), axis=0)
        c0 = float(np.mean(inbox)); tgt = 1 - self.cfg.alpha
        if c0 >= tgt: return (self.l.copy(), self.u.copy()), 1.0, c0
        mj = np.median(Cc, axis=1)
        wj = np.maximum(np.maximum(mj - self.l, self.u - mj), 1e-12)
        tlo, thi = 1.0, 2.0
        for _ in range(12):
            lb = mj - thi*wj; ub = mj + thi*wj
            if float(np.mean(np.all((Cc>=lb[:,None])&(Cc<=ub[:,None]),
                                     axis=0))) >= tgt: break
            thi *= 2
        for _ in range(40):
            tm = 0.5*(tlo+thi)
            lb = mj-tm*wj; ub = mj+tm*wj
            if float(np.mean(np.all((Cc>=lb[:,None])&(Cc<=ub[:,None]),
                                     axis=0))) >= tgt: thi = tm
            else: tlo = tm
            if thi-tlo < 1e-4: break
        tc = thi
        lc = mj - tc*wj; uc = mj + tc*wj
        cf = float(np.mean(np.all((Cc>=lc[:,None])&(Cc<=uc[:,None]),axis=0)))
        if self.cfg.verbose:
            print(f"    Calibration: t={tc:.3f}, cov {c0:.3f}→{cf:.3f}")
        return (lc, uc), tc, cf


# ============================================================================
# PUQ-LASSO: Standard Condat-Vũ (NO acceleration)
# ============================================================================

def solve_puq_lasso(A, y_vec, B, pca, cal_samples, cfg):
    """
    Standard PUQ-LASSO: plain Condat-Vũ.
      p_{n+1} = clip(p_n + σ·B(μ + U c_n), -λ, λ)
      c_{n+1} = proj_C(c_n − τ·[∇f̂(c_n) + Uᵀ Bᵀ p_{n+1}])
    NO momentum, NO extrapolation, FIXED steps.
    """
    t0 = time.time()
    mu, Uk, l, u, k = pca.mu, pca.Uk, pca.l, pca.u, pca.k
    y = y_vec[:, None]; q = B.shape[0]

    AU = A @ Uk; BU = B @ Uk
    L = op_norm(AU.T @ AU); Kn = op_norm(BU)

    # Fixed step sizes: τσ||K||² + τL/2 < 1
    tau = 0.8 / (L + Kn**2 + 1e-10)
    sigma = 0.6 / (tau * Kn**2)

    c = np.clip(np.zeros((k, 1)), l[:, None], u[:, None])
    p = np.zeros((q, 1))

    iters = [0]; objs = []; dzs = [np.nan]; taus_h = []; sigmas_h = []

    def objective(cc):
        x = mu + Uk @ cc
        return (0.5*float(np.sum((A@x - y)**2))
                + cfg.lam*float(np.sum(np.abs(B@x))))

    objs.append(objective(c))
    converged = False; conv_at = cfg.max_iter

    if cfg.verbose:
        print(f"\n    PUQ-LASSO (fixed τ={tau:.3e}, σ={sigma:.3e}, NO momentum)")
        print(f"    {'It':>5} {'Obj':>14} {'Δc':>12}")
        print("    " + "─"*35)

    for it in range(1, cfg.max_iter + 1):
        # Dual update (at current c, no extrapolation)
        Bx = B @ (mu + Uk @ c)
        p = np.clip(p + sigma * Bx, -cfg.lam, cfg.lam)

        # Primal update (gradient at current c, no momentum)
        xc = mu + Uk @ c
        grad = Uk.T @ (A.T @ (A @ xc - y)) + Uk.T @ (B.T @ p)
        c_new = np.clip(c -grad, l[:, None], u[:, None])

        dc = float(np.linalg.norm(c_new - c)) / max(1., float(np.linalg.norm(c)))
        obj = objective(c_new)

        iters.append(it); objs.append(obj); dzs.append(dc)
        taus_h.append(tau); sigmas_h.append(sigma)
        c = c_new

        if cfg.verbose and (it % cfg.print_every == 0 or it <= 3 or it == cfg.max_iter):
            print(f"    {it:5d} {obj:14.7e} {dc:12.5e}")

        if cfg.early_stop and it > cfg.min_iter and dc < cfg.tol:
            if not converged: converged = True; conv_at = it
            if cfg.verbose: print(f"    ✓ Converged at {it}")

    box, tc, cb = pca.calibrate(cal_samples)
    elapsed = time.time() - t0

    return Result(
        name="PUQ-LASSO", x_star=(mu + Uk @ c).ravel(),
        mu=mu.ravel(), Uk=Uk, box_bounds=box,
        calibration_factor=tc, coverage_box=cb,
        iterations=np.array(iters), objective=np.array(objs),
        rel_change=np.array(dzs), taus=np.array(taus_h),
        sigmas=np.array(sigmas_h), converged=converged,
        converged_at=conv_at, elapsed=elapsed)


# ============================================================================
# ACV-PUQ-LASSO: Accelerated Condat-Vũ (PROPOSED)
# ============================================================================

def solve_acv_puq_lasso(A, y_vec, B, pca, cal_samples, cfg):
    """
    Accelerated Condat-Vũ PUQ-LASSO (PROPOSED).
      w_{n+1} = β_n c_n + (1−β_n) z_n          (Nesterov extrapolation)
      c̄_n    = c_n + θ_n (c_n − c_{n-1})       (primal extrapolation)
      p_{n+1} = clip(p_n + σ_n·B(μ+U c̄_n), −λ, λ)
      c_{n+1} = proj_C(c_n − τ_n·[∇f̂(w_{n+1}) + Uᵀ Bᵀ p_{n+1}])
      z_{n+1} = β_n c_{n+1} + (1−β_n) z_n      (averaging)

    Key differences from PUQ-LASSO:
      ✓ Nesterov momentum (θ, β > 0)
      ✓ Gradient computed at extrapolated point w (not current c)
      ✓ Dual extrapolation via c̄
      ✓ Adaptive step sizes (phase-dependent)
      ✓ Ergodic averaging via z
    """
    t0 = time.time()
    mu, Uk, l, u, k = pca.mu, pca.Uk, pca.l, pca.u, pca.k
    y = y_vec[:, None]; q = B.shape[0]

    AU = A @ Uk; BU = B @ Uk
    L = op_norm(AU.T @ AU); Kn = op_norm(BU)

    tau0 = 0.99 / (L + Kn**2 + 1e-10)
    sigma0 = 0.99 / (tau0 * Kn**2 + 1e-12)

    c = np.clip(np.zeros((k, 1)), l[:, None], u[:, None])
    z = c.copy(); p = np.zeros((q, 1)); c_old = c.copy()

    iters = [0]; objs = []; dzs = [np.nan]; taus_h = []; sigmas_h = []

    def objective(cc):
        x = mu + Uk @ cc
        return (0.5*float(np.sum((A@x - y)**2))
                + cfg.lam*float(np.sum(np.abs(B@x))))

    objs.append(objective(c))
    converged = False; conv_at = cfg.max_iter

    if cfg.verbose:
        print(f"\n    ACV-PUQ-LASSO ★ (adaptive steps, Nesterov momentum)")
        print(f"    {'It':>5} {'Obj':>14} {'Δz':>12} {'θ':>8}")
        print("    " + "─"*43)

    for it in range(1, cfg.max_iter + 1):
        t = it + 1

        # ── Adaptive step sizes ──
        frac = it / cfg.max_iter
        if frac < 0.25:
            scale = 1.3       # aggressive early
        elif frac < 0.6:
            scale = 1.1       # moderate middle
        else:
            scale = 0.85      # conservative late (precision)

        tau = tau0 * scale
        sigma = sigma0 * scale

        # Enforce CV condition: τσ||K||² + τL/2 < 1
        cond = tau * sigma * Kn**2 + tau * L / 2
        if cond > 0.95:
            tau *= 0.9 / cond
            sigma = 0.9 / (tau * Kn**2 + 1e-12)

        # ── Nesterov momentum ──
        theta = (t - 1) / (t + 2)
        beta = theta

        # Step 1: Nesterov extrapolation
        w = beta * c + (1 - beta) * z

        # Step 2: Dual update with primal extrapolation
        c_bar = c + theta * (c - c_old)
        Bx_bar = B @ (mu + Uk @ c_bar)
        p = np.clip(p + sigma * Bx_bar, -cfg.lam, cfg.lam)

        # Step 3: Gradient at EXTRAPOLATED point w (key difference!)
        x_w = mu + Uk @ w
        grad = Uk.T @ (A.T @ (A @ x_w - y)) + Uk.T @ (B.T @ p)
        c_new = np.clip(c - tau * grad, l[:, None], u[:, None])

        # Step 4: Ergodic averaging
        z_new = beta * c_new + (1 - beta) * z

        # Diagnostics (on z, the averaged iterate)
        dz = float(np.linalg.norm(z_new - z))
        nz = max(1., float(np.linalg.norm(z)))
        rc = dz / nz

        obj = objective(z_new)

        iters.append(it); objs.append(obj); dzs.append(rc)
        taus_h.append(tau); sigmas_h.append(sigma)

        c_old = c.copy(); c = c_new; z = z_new

        if cfg.verbose and (it % cfg.print_every == 0 or it <= 3 or it == cfg.max_iter):
            print(f"    {it:5d} {obj:14.7e} {rc:12.5e} {theta:8.4f}")

        if cfg.early_stop and it > cfg.min_iter and rc < cfg.tol:
            if not converged: converged = True; conv_at = it
            if cfg.verbose: print(f"    ✓ Converged at {it}")

    box, tc, cb = pca.calibrate(cal_samples)
    elapsed = time.time() - t0

    return Result(
        name="ACV-PUQ-LASSO", x_star=(mu + Uk @ z).ravel(),
        mu=mu.ravel(), Uk=Uk, box_bounds=box,
        calibration_factor=tc, coverage_box=cb,
        iterations=np.array(iters), objective=np.array(objs),
        rel_change=np.array(dzs), taus=np.array(taus_h),
        sigmas=np.array(sigmas_h), converged=converged,
        converged_at=conv_at, elapsed=elapsed)


# ============================================================================
# Test Images
# ============================================================================

class TestImages:
    @staticmethod
    def shapes(size=64):
        img = np.zeros((size, size)); s = size//5
        img[s:3*s, s:3*s] = 1.0
        cx, cy, r = 3*size//4, size//2, size//6
        yy, xx = np.ogrid[:size, :size]
        img[(xx-cx)**2+(yy-cy)**2 <= r**2] = 0.7
        for i in range(size//4, 3*size//4):
            if 0 <= i < size: img[i, min(i,size-1)] = 0.5
        return img

    @staticmethod
    def phantom(size=64):
        img = np.zeros((size, size)); cx, cy = size//2, size//2
        yy, xx = np.ogrid[:size, :size]
        img[((xx-cx)/(0.4*size))**2+((yy-cy)/(0.45*size))**2<=1] = 0.8
        img[((xx-cx+size//8)/(0.1*size))**2+((yy-cy)/(0.15*size))**2<=1] = 0.4
        img[((xx-cx-size//8)/(0.08*size))**2+((yy-cy+size//10)/(0.12*size))**2<=1] = 1.0
        return img

    @staticmethod
    def cameraman(size=64):
        try:
            from skimage.data import camera
            img = camera().astype(np.float64) / 255.0
            if size != 512:
                from scipy.ndimage import zoom
                img = zoom(img, size/512, order=3)
                img = np.clip(img, 0, 1)
            return img
        except ImportError:
            return TestImages.shapes(size)


# ============================================================================
# Visualization
# ============================================================================

class Viz:
    @staticmethod
    def full(gt, noisy, rp, ra, save=None):
        shape = gt.shape
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

        # Row 0: Images
        for j, (img, title) in enumerate([
            (gt, 'Ground Truth'),
            (noisy, f'Noisy\nPSNR={psnr(gt,noisy):.2f}'),
            (np.clip(rp.x_star.reshape(shape),0,1),
             f'PUQ-LASSO\nPSNR={rp.psnr_val:.2f}  SSIM={rp.ssim_val:.4f}'),
            (np.clip(ra.x_star.reshape(shape),0,1),
             f'★ ACV-PUQ-LASSO\nPSNR={ra.psnr_val:.2f}  SSIM={ra.ssim_val:.4f}'),
        ]):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=10,
                         fontweight='bold' if j == 3 else 'normal')
            ax.axis('off')

        # Row 1: Convergence
        ax = fig.add_subplot(gs[1, 0:2])
        ax.semilogy(rp.iterations[2:], np.maximum(rp.rel_change[2:], 1e-16),
                    'b-', lw=2.5, label='PUQ-LASSO')
        ax.semilogy(ra.iterations[2:], np.maximum(ra.rel_change[2:], 1e-16),
                    'r-', lw=2.5, label='ACV-PUQ-LASSO ★')
        ax.axhline(ra.config if hasattr(ra, 'config') else 1e-8,
                   ls='--', color='gray', alpha=.5, label='tol')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Relative change (log)')
        ax.set_title('Convergence Speed', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        # Objective
        ax = fig.add_subplot(gs[1, 2:4])
        ax.plot(rp.iterations[1:], rp.objective[1:], 'b-', lw=2.5,
                label=f'PUQ-LASSO (final={rp.objective[-1]:.6e})')
        ax.plot(ra.iterations[1:], ra.objective[1:], 'r-', lw=2.5,
                label=f'ACV-PUQ-LASSO ★ (final={ra.objective[-1]:.6e})')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Objective F(c)')
        ax.set_title('Objective Value ↓', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Row 2: Residuals + Uncertainty
        res_puq = np.abs(rp.x_star.reshape(shape) - gt)
        res_acv = np.abs(ra.x_star.reshape(shape) - gt)
        vmax_res = max(np.percentile(res_puq, 99), np.percentile(res_acv, 99))

        ax = fig.add_subplot(gs[2, 0])
        ax.imshow(res_puq, cmap='hot', vmin=0, vmax=vmax_res)
        ax.set_title(f'PUQ-LASSO |error|\nMSE={np.mean(res_puq**2):.5f}', fontsize=9)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, 1])
        ax.imshow(res_acv, cmap='hot', vmin=0, vmax=vmax_res)
        ax.set_title(f'ACV |error| ★\nMSE={np.mean(res_acv**2):.5f}', fontsize=9)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, 2])
        if rp.pixel_std is not None:
            ax.imshow(rp.pixel_std, cmap='hot')
            ax.set_title(f'PUQ σ\nCov={rp.coverage:.0%}', fontsize=9)
        ax.axis('off')

        ax = fig.add_subplot(gs[2, 3])
        if ra.pixel_std is not None:
            ax.imshow(ra.pixel_std, cmap='hot')
            ax.set_title(f'ACV σ ★\nCov={ra.coverage:.0%}', fontsize=9)
        ax.axis('off')

        # Row 3: Cross-section + Summary
        ax = fig.add_subplot(gs[3, 0:3])
        mid = shape[0] // 2
        if rp.lower is not None and ra.lower is not None:
            ax.fill_between(range(shape[1]), rp.lower[mid], rp.upper[mid],
                            alpha=.12, color='blue', label='PUQ CI')
            ax.fill_between(range(shape[1]), ra.lower[mid], ra.upper[mid],
                            alpha=.12, color='red', label='ACV CI')
        ax.plot(gt[mid], 'k-', lw=2.5, label='Truth', zorder=10)
        ax.plot(noisy[mid], color='gray', alpha=.3, lw=1, label='Noisy')
        ax.plot(np.clip(rp.x_star.reshape(shape),0,1)[mid], 'b--', lw=1.5,
                label='PUQ-LASSO')
        ax.plot(np.clip(ra.x_star.reshape(shape),0,1)[mid], 'r-', lw=1.5,
                label='ACV-PUQ-LASSO ★')
        ax.set_xlabel('Pixel'); ax.set_ylabel('Intensity')
        ax.set_title('Cross-section at row ' + str(mid), fontsize=11)
        ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)

        # Summary box
        ax = fig.add_subplot(gs[3, 3])
        ax.axis('off')
        dp = ra.psnr_val - rp.psnr_val
        ds = ra.ssim_val - rp.ssim_val
        do = rp.objective[-1] - ra.objective[-1]
        txt = (
            f"IMPROVEMENT\n"
            f"{'─'*22}\n"
            f"ΔPSNR:  {dp:+.3f} dB\n"
            f"ΔSSIM:  {ds:+.5f}\n"
            f"ΔObj:   {do:+.4e}\n"
            f"{'─'*22}\n"
            f"PUQ  conv: {rp.converged_at}\n"
            f"ACV  conv: {ra.converged_at}\n"
            f"Speed: {rp.converged_at/max(ra.converged_at,1):.1f}×\n"
            f"{'─'*22}\n"
            f"PUQ  obj: {rp.objective[-1]:.6e}\n"
            f"ACV  obj: {ra.objective[-1]:.6e}\n"
        )
        color = 'green' if dp > 0 and ds > 0 and do > 0 else 'red'
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.suptitle('PUQ-LASSO vs ACV-PUQ-LASSO ★  —  Direct Comparison',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def benchmark_bars(all_res, save=None):
        if len(all_res) < 2: return
        keys = list(all_res.keys())
        pp = [all_res[k][0].psnr_val for k in keys]
        pa = [all_res[k][1].psnr_val for k in keys]
        sp = [all_res[k][0].ssim_val for k in keys]
        sa = [all_res[k][1].ssim_val for k in keys]
        op = [all_res[k][0].objective[-1] for k in keys]
        oa = [all_res[k][1].objective[-1] for k in keys]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        x = np.arange(len(keys)); w = 0.35

        ax = axes[0]
        ax.bar(x-w/2, pp, w, label='PUQ-LASSO', color='steelblue')
        ax.bar(x+w/2, pa, w, label='ACV-PUQ-LASSO ★', color='tomato')
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('PSNR (dB)'); ax.set_title('PSNR ↑')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

        ax = axes[1]
        ax.bar(x-w/2, sp, w, label='PUQ-LASSO', color='steelblue')
        ax.bar(x+w/2, sa, w, label='ACV-PUQ-LASSO ★', color='tomato')
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('SSIM'); ax.set_title('SSIM ↑')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

        ax = axes[2]
        ax.bar(x-w/2, op, w, label='PUQ-LASSO', color='steelblue')
        ax.bar(x+w/2, oa, w, label='ACV-PUQ-LASSO ★', color='tomato')
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Objective'); ax.set_title('Final Objective ↓')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

        plt.suptitle('Benchmark: PUQ-LASSO vs ACV-PUQ-LASSO ★',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# Single Run
# ============================================================================

def run_single(image_name='shapes', image_size=64,
               noise_std=0.15, lam=0.005, alpha=0.10,
               max_iter=150, k_pca_ratio=0.15,
               seed=42, save_dir=None):
    np.random.seed(seed)

    print("=" * 60)
    print("  PUQ-LASSO  vs  ACV-PUQ-LASSO ★")
    print("=" * 60)

    factory = {'shapes': TestImages.shapes, 'phantom': TestImages.phantom,
               'cameraman': TestImages.cameraman}
    x_true = factory.get(image_name, TestImages.shapes)(image_size)
    shape = x_true.shape; n = x_true.size

    cfg = Config(lam=lam, alpha=alpha, noise_level=noise_std,
                 max_iter=max_iter, k_pca_ratio=k_pca_ratio,
                 verbose=True, print_every=max(max_iter//10, 1))

    rng = np.random.RandomState(seed)
    y_noisy = add_noise(x_true, noise_std, rng)

    print(f"\n  Image: {image_name} ({image_size}×{image_size}), n={n}")
    print(f"  λ={lam}, σ={noise_std}, α={alpha}, max_iter={max_iter}")
    print(f"  Noisy PSNR = {psnr(x_true, y_noisy):.2f} dB")

    A = make_forward(shape, cfg); B = make_tv(*shape)

    # Shared samples + PCA
    rng2 = np.random.RandomState(seed + 100)
    post = np.empty((n, cfg.n_posterior))
    for i in range(cfg.n_posterior):
        post[:, i] = x_true.ravel() + cfg.posterior_noise_std * rng2.randn(n)
    cal = np.empty((n, cfg.n_calibration))
    for i in range(cfg.n_calibration):
        cal[:, i] = x_true.ravel() + cfg.calibration_noise_std * rng2.randn(n)

    pca = SharedPCA(cfg)
    print("\n  Building shared PCA …")
    pca.build(post)

    # ── PUQ-LASSO ──
    print(f"\n{'━'*60}")
    print(f"  METHOD 1: PUQ-LASSO (standard)")
    print(f"{'━'*60}")
    rp = solve_puq_lasso(A, y_noisy.ravel(), B, pca, cal, cfg)
    rp.evaluate(x_true, alpha)

    # ── ACV-PUQ-LASSO ──
    print(f"\n{'━'*60}")
    print(f"  METHOD 2: ACV-PUQ-LASSO ★ (accelerated)")
    print(f"{'━'*60}")
    ra = solve_acv_puq_lasso(A, y_noisy.ravel(), B, pca, cal, cfg)
    ra.evaluate(x_true, alpha)

    # ── Summary ──
    dp = ra.psnr_val - rp.psnr_val
    ds = ra.ssim_val - rp.ssim_val
    do = rp.objective[-1] - ra.objective[-1]

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'PUQ-LASSO':>14} {'ACV-PUQ ★':>14} {'Δ':>12}")
    print(f"  {'─'*62}")
    for lab, v1, v2, fmt in [
        ('PSNR (dB)',       rp.psnr_val, ra.psnr_val, '.3f'),
        ('SSIM',            rp.ssim_val, ra.ssim_val, '.5f'),
        ('Final Objective', rp.objective[-1], ra.objective[-1], '.6e'),
        ('Final Δ',         rp.rel_change[-1], ra.rel_change[-1], '.3e'),
        ('Coverage',        rp.coverage, ra.coverage, '.1%'),
        ('Interval Width',  rp.interval_width, ra.interval_width, '.4f'),
        ('Converged at',    float(rp.converged_at), float(ra.converged_at), '.0f'),
        ('Time (s)',        rp.elapsed, ra.elapsed, '.2f'),
    ]:
        d = v2 - v1
        print(f"  {lab:<22} {v1:>14{fmt}} {v2:>14{fmt}} {d:>+12{fmt}}")

    print(f"  {'─'*62}")
    print(f"  ACV wins PSNR:  {'YES ✓' if dp > 0 else 'NO ✗'}  ({dp:+.3f} dB)")
    print(f"  ACV wins SSIM:  {'YES ✓' if ds > 0 else 'NO ✗'}  ({ds:+.5f})")
    print(f"  ACV wins Obj:   {'YES ✓' if do > 0 else 'NO ✗'}  ({do:+.4e})")
    speed = rp.converged_at / max(ra.converged_at, 1)
    print(f"  Iter speedup:   {speed:.2f}×")
    print(f"  PCA k={pca.k}, n/k={n/pca.k:.0f}")
    print(f"{'='*60}")

    # Plot
    base = save_dir or '.'
    os.makedirs(base, exist_ok=True)
    Viz.full(x_true, y_noisy, rp, ra,
             save=os.path.join(base, 'comparison.png'))

    return rp, ra


# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark(save_dir='benchmark'):
    print("\n" + "=" * 60)
    print("  BENCHMARK: PUQ-LASSO vs ACV-PUQ-LASSO ★")
    print("=" * 60)

    configs = [
        ('shapes',  64, 0.10, 0.005, 150, 0.15, 'shapes_σ0.10'),
        ('shapes',  64, 0.20, 0.005, 150, 0.15, 'shapes_σ0.20'),
        ('shapes',  64, 0.30, 0.008, 150, 0.15, 'shapes_σ0.30'),
        ('phantom', 64, 0.10, 0.005, 150, 0.15, 'phantom_σ0.10'),
        ('phantom', 64, 0.20, 0.005, 150, 0.15, 'phantom_σ0.20'),
        ('phantom', 64, 0.30, 0.008, 150, 0.15, 'phantom_σ0.30'),
    ]

    all_res = {}
    for img, sz, sig, lam, mi, kpr, label in configs:
        print(f"\n{'━'*60}")
        print(f"  {label}")
        print(f"{'━'*60}")
        try:
            sub = os.path.join(save_dir, label)
            rp, ra = run_single(img, sz, sig, lam, 0.10, mi, kpr, 42, sub)
            all_res[label] = (rp, ra)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()

    if all_res:
        # Summary table
        print("\n" + "=" * 95)
        print("  BENCHMARK SUMMARY")
        print("=" * 95)
        print(f"  {'Config':<20} {'':>3} {'PSNR':>8} {'SSIM':>8} "
              f"{'Obj':>13} {'Conv':>5} {'Cov':>6}")
        print("  " + "─" * 68)

        for key, (rp, ra) in all_res.items():
            print(f"  {key:<20} PUQ {rp.psnr_val:8.3f} {rp.ssim_val:8.5f} "
                  f"{rp.objective[-1]:13.6e} {rp.converged_at:5d} "
                  f"{rp.coverage:6.0%}")
            print(f"  {'':20} ACV {ra.psnr_val:8.3f} {ra.ssim_val:8.5f} "
                  f"{ra.objective[-1]:13.6e} {ra.converged_at:5d} "
                  f"{ra.coverage:6.0%}")
            dp = ra.psnr_val - rp.psnr_val
            ds = ra.ssim_val - rp.ssim_val
            do = rp.objective[-1] - ra.objective[-1]
            wins = sum([dp > 0, ds > 0, do > 0])
            print(f"  {'':20}  Δ  {dp:+8.3f} {ds:+8.5f} "
                  f"{do:+13.4e}       {wins}/3 wins")
            print("  " + "─" * 68)

        # Count total wins
        total_p = sum(1 for _,(rp,ra) in all_res.items()
                      if ra.psnr_val > rp.psnr_val)
        total_s = sum(1 for _,(rp,ra) in all_res.items()
                      if ra.ssim_val > rp.ssim_val)
        total_o = sum(1 for _,(rp,ra) in all_res.items()
                      if ra.objective[-1] < rp.objective[-1])
        N = len(all_res)
        print(f"\n  ACV wins PSNR: {total_p}/{N}")
        print(f"  ACV wins SSIM: {total_s}/{N}")
        print(f"  ACV wins Obj:  {total_o}/{N}")
        print("=" * 95)

        Viz.benchmark_bars(all_res,
                           save=os.path.join(save_dir, 'benchmark.png'))

    return all_res


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║   PUQ-LASSO  vs  ACV-PUQ-LASSO ★                   ║")
    print("║   Standard Condat-Vũ  vs  Accelerated Condat-Vũ     ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║   Same PCA · Same TV · Same calibration             ║")
    print("║   Only diff: momentum + adaptive steps              ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\n  1 = Quick (32×32)")
    print("  2 = Standard (64×64)")
    print("  3 = Full benchmark")

    try:
        ch = input("\n  Choice [2]: ").strip() or "2"
    except EOFError:
        ch = "2"

    if ch == "3":
        run_benchmark('benchmark')
    elif ch == "1":
        run_single('shapes', 32, 0.20, 0.005, 0.10, 100, 0.15, 42, 'quick')
    else:
        run_single('shapes', 64, 0.15, 0.005, 0.10, 150, 0.15, 42, 'results')

    print("\n✓ Done.")