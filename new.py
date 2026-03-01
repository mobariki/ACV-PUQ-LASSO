import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from scipy.ndimage import convolve
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import norm as sparse_norm
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def lasso_objective(A, y, x, lam):
    """F(x) = 0.5‖Ax − y‖² + λ‖x‖₁."""
    residual = A @ x - y
    return 0.5 * np.dot(residual, residual) + lam * np.sum(np.abs(x))


def relative_error(x_est, x_true):
    """‖x_est − x_true‖₂ / ‖x_true‖₂."""
    denom = np.linalg.norm(x_true)
    if denom < 1e-15:
        return np.linalg.norm(x_est)
    return np.linalg.norm(x_est - x_true) / denom


def psnr(img_true, img_est):
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((img_true - img_est) ** 2)
    if mse < 1e-15:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_simple(img_true, img_est, win_size=7):
    """Simplified SSIM (structural similarity)."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    kernel = np.ones((win_size, win_size)) / (win_size ** 2)

    mu1 = convolve(img_true, kernel, mode='reflect')
    mu2 = convolve(img_est, kernel, mode='reflect')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = convolve(img_true ** 2, kernel, mode='reflect') - mu1_sq
    sigma2_sq = convolve(img_est ** 2, kernel, mode='reflect') - mu2_sq
    sigma12 = convolve(img_true * img_est, kernel, mode='reflect') - mu12

    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return np.mean(num / den)


def support_recovery_f1(x_est, x_true, tol=1e-3):
    """F1-score for support recovery."""
    threshold = tol * np.max(np.abs(x_true)) if np.max(np.abs(x_true)) > 0 else tol
    true_support = np.abs(x_true) > threshold
    est_support = np.abs(x_est) > threshold
    tp = np.sum(true_support & est_support)
    fp = np.sum(~true_support & est_support)
    fn = np.sum(true_support & ~est_support)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def build_ill_conditioned_matrix(m, n, cond_number=100.0):
    """Build m×n matrix with prescribed condition number."""
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    sigma_max = 10.0
    sigma_min = sigma_max / cond_number
    singular_values = np.geomspace(sigma_max, sigma_min, min(m, n))
    S = np.zeros((m, n))
    np.fill_diagonal(S, singular_values)
    return U @ S @ V.T


def build_correlated_matrix(m, n, rho=0.8):
    """Build measurement matrix with correlated columns."""
    Z = np.random.randn(m, n)
    A = Z.copy()
    for j in range(1, n):
        A[:, j] = Z[:, j] + rho * Z[:, j - 1]
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    A /= (col_norms + 1e-12)
    return A * np.sqrt(m)


def adaptive_lambda(lam_base, noise_std, n, m):
    """λ(σ) = λ_base + C·σ·√(log(n)/m)."""
    if noise_std < 1e-12:
        return lam_base
    C = 0.5
    return lam_base + C * noise_std * np.sqrt(np.log(n) / m)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER: PUQ-LASSO (standard Condat–Vũ)
# ═══════════════════════════════════════════════════════════════════════════════

def run_puq_lasso(A, y, x_true, lam, N, record_every=10):
    """Standard Condat–Vũ splitting."""
    m, n = A.shape
    L = np.linalg.norm(A, 2) ** 2
    K_norm = 1.0

    tau = 0.99 / (L + K_norm ** 2)
    sigma = 0.99 / (tau * K_norm ** 2 + 1e-12)

    x = np.zeros(n)
    p = np.zeros(n)

    objectives = []
    rel_errors = []
    record_iters = []

    for k in range(1, N + 1):
        p = np.clip(p + sigma * x, -lam, lam)
        grad_f = A.T @ (A @ x - y)
        x = x - tau * (grad_f + p)

        if k % record_every == 0 or k == N or k == 1:
            objectives.append(lasso_objective(A, y, x, lam))
            rel_errors.append(relative_error(x, x_true))
            record_iters.append(k)

    return x, np.array(objectives), np.array(rel_errors), np.array(record_iters)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER: ACV-PUQ-LASSO (Accelerated Condat–Vũ)
# ═══════════════════════════════════════════════════════════════════════════════

def run_acv_puq_lasso(A, y, x_true, lam, N, record_every=10):
    """Accelerated Condat–Vũ with Nesterov momentum."""
    m, n = A.shape
    L = np.linalg.norm(A, 2) ** 2
    K_norm = 1.0

    tau0 = 0.99 / (L + K_norm ** 2)
    sigma0 = 0.99 / (tau0 * K_norm ** 2 + 1e-12)

    x = np.zeros(n)
    x_old = np.zeros(n)
    p = np.zeros(n)
    z_sum = np.zeros(n)
    weight_sum = 0.0

    objectives = []
    rel_errors = []
    record_iters = []

    for k in range(1, N + 1):
        frac = k / N
        if frac < 0.3:
            scale = 1.3
        elif frac < 0.7:
            scale = 1.0
        else:
            scale = 0.8

        tau = tau0 * scale
        sigma = sigma0 * scale

        cond_val = tau * sigma * K_norm ** 2 + tau * L / 2.0
        if cond_val >= 0.95:
            damp = np.sqrt(0.95 / cond_val)
            tau *= damp
            sigma *= damp

        theta = (k - 1.0) / (k + 2.0)
        w = x + theta * (x - x_old)

        p = np.clip(p + sigma * w, -lam, lam)
        grad_f = A.T @ (A @ w - y)
        x_new = w - tau * (grad_f + p)

        wt = float(k)
        weight_sum += wt
        z_sum = z_sum + wt * x_new
        z_out = z_sum / weight_sum

        x_old = x.copy()
        x = x_new

        if k % record_every == 0 or k == N or k == 1:
            objectives.append(lasso_objective(A, y, z_out, lam))
            rel_errors.append(relative_error(z_out, x_true))
            record_iters.append(k)

    return z_out, np.array(objectives), np.array(rel_errors), np.array(record_iters)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE DENOISING SOLVERS (matrix-free for large images)
# ═══════════════════════════════════════════════════════════════════════════════

def finite_diff_2d(img):
    """Compute discrete gradient (forward differences) of 2D image."""
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, :-1] = img[:, 1:] - img[:, :-1]
    dy[:-1, :] = img[1:, :] - img[:-1, :]
    return dx, dy


def finite_diff_2d_transpose(dx, dy):
    """Adjoint of discrete gradient (backward differences)."""
    M, N = dx.shape
    div = np.zeros((M, N))
    # d/dx transpose
    div[:, 0] = -dx[:, 0]
    div[:, 1:-1] = dx[:, :-2] - dx[:, 1:-1]
    div[:, -1] = dx[:, -2]
    # d/dy transpose
    div[0, :] -= dy[0, :]
    div[1:-1, :] += dy[:-2, :] - dy[1:-1, :]
    div[-1, :] += dy[-2, :]
    return div


def run_puq_image_denoise(noisy_img, clean_img, lam, N, record_every=50):
    """
    PUQ-LASSO for image denoising via analysis-based ℓ1.
    
    min_x  0.5‖x − y‖² + λ‖∇x‖₁  (anisotropic TV)
    
    Primal-dual with K = ∇ (finite differences):
        p^{k+1} = clip(p^k + σ∇x^k, −λ, λ)
        x^{k+1} = x^k − τ((x^k − y) − div(p^{k+1}))
    """
    M, N_cols = noisy_img.shape
    y = noisy_img.copy()
    x = y.copy()

    # K = ∇, ‖K‖ = √8 for 2D finite differences
    L = 1.0  # Lipschitz of ∇f = x − y
    K_norm_sq = 8.0

    tau = 0.99 / (L + K_norm_sq)
    sigma = 0.99 / (tau * K_norm_sq + 1e-12)

    # Dual variables (two components for dx, dy)
    px = np.zeros_like(x)
    py = np.zeros_like(x)

    psnr_vals = []
    ssim_vals = []
    record_iters = []

    for k in range(1, N + 1):
        # Gradient of x
        dx, dy = finite_diff_2d(x)

        # Dual update
        px = np.clip(px + sigma * dx, -lam, lam)
        py = np.clip(py + sigma * dy, -lam, lam)

        # Divergence (adjoint of gradient)
        div_p = finite_diff_2d_transpose(px, py)

        # Primal update: gradient of 0.5‖x−y‖² is (x−y)
        x = x - tau * ((x - y) - div_p)

        # Clamp to [0, 1]
        x = np.clip(x, 0, 1)

        if k % record_every == 0 or k == N or k == 1:
            psnr_vals.append(psnr(clean_img, x))
            ssim_vals.append(ssim_simple(clean_img, x))
            record_iters.append(k)

    return x, np.array(psnr_vals), np.array(ssim_vals), np.array(record_iters)


def run_acv_image_denoise(noisy_img, clean_img, lam, N, record_every=50):
    """
    ACV-PUQ-LASSO for image denoising via analysis-based ℓ1.
    
    Accelerated primal-dual with:
      1. Nesterov momentum on primal variable
      2. Three-phase adaptive step sizes
      3. Ergodic averaging
    """
    M, N_cols = noisy_img.shape
    y = noisy_img.copy()
    x = y.copy()
    x_old = x.copy()

    L = 1.0
    K_norm_sq = 8.0

    tau0 = 0.99 / (L + K_norm_sq)
    sigma0 = 0.99 / (tau0 * K_norm_sq + 1e-12)

    px = np.zeros_like(x)
    py = np.zeros_like(x)

    # Ergodic averaging
    z_sum = np.zeros_like(x)
    weight_sum = 0.0

    psnr_vals = []
    ssim_vals = []
    record_iters = []

    for k in range(1, N + 1):
        frac = k / N
        if frac < 0.3:
            scale = 1.3
        elif frac < 0.7:
            scale = 1.0
        else:
            scale = 0.8

        tau = tau0 * scale
        sigma = sigma0 * scale

        cond_val = tau * sigma * K_norm_sq + tau * L / 2.0
        if cond_val >= 0.95:
            damp = np.sqrt(0.95 / cond_val)
            tau *= damp
            sigma *= damp

        # Nesterov extrapolation
        theta = (k - 1.0) / (k + 2.0)
        w = x + theta * (x - x_old)

        # Gradient of w
        dx, dy = finite_diff_2d(w)

        # Dual update
        px = np.clip(px + sigma * dx, -lam, lam)
        py = np.clip(py + sigma * dy, -lam, lam)

        # Divergence
        div_p = finite_diff_2d_transpose(px, py)

        # Primal update from extrapolated point
        x_new = w - tau * ((w - y) - div_p)
        x_new = np.clip(x_new, 0, 1)

        # Ergodic averaging
        wt = float(k)
        weight_sum += wt
        z_sum = z_sum + wt * x_new
        z_out = z_sum / weight_sum
        z_out = np.clip(z_out, 0, 1)

        x_old = x.copy()
        x = x_new

        if k % record_every == 0 or k == N or k == 1:
            psnr_vals.append(psnr(clean_img, z_out))
            ssim_vals.append(ssim_simple(clean_img, z_out))
            record_iters.append(k)

    return z_out, np.array(psnr_vals), np.array(ssim_vals), np.array(record_iters)


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _print_results(tid, x_puq, x_acv, x_true, err_puq, err_acv,
                   obj_puq, obj_acv, A, N):
    m, n = A.shape
    print(f"  Condition number κ(A^T A) ≈ {np.linalg.cond(A.T @ A):.2e}")
    print(f"  Iterations: {N}")
    print()
    print(f"  PUQ-LASSO      — rel error: {err_puq[-1]:.6f}, "
          f"objective: {obj_puq[-1]:.6e}")
    print(f"  ACV-PUQ-LASSO  — rel error: {err_acv[-1]:.6f}, "
          f"objective: {obj_acv[-1]:.6e}")
    print(f"  Improvement    — Δ rel err: {err_puq[-1] - err_acv[-1]:+.6f}, "
          f"Δ obj: {obj_puq[-1] - obj_acv[-1]:+.4e}")

    f1_p, pr_p, re_p = support_recovery_f1(x_puq, x_true)
    f1_a, pr_a, re_a = support_recovery_f1(x_acv, x_true)
    print(f"  Support F1     — PUQ: {f1_p:.3f} (P={pr_p:.2f}, R={re_p:.2f})")
    print(f"                 — ACV: {f1_a:.3f} (P={pr_a:.2f}, R={re_a:.2f})")


# ═══════════════════════════════════════════════════════════════════════════════
# SEPARATE FIGURE SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_test_figures(obj_puq, obj_acv, err_puq, err_acv,
                      iters_puq, iters_acv,
                      x_puq, x_acv, x_true, test_name, test_id):
    """Save three separate figures per sparse recovery test."""

    # ── Objective ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(iters_puq, obj_puq, 'b-', lw=2, alpha=0.85, label='PUQ-LASSO')
    ax.semilogy(iters_acv, obj_acv, 'r--', lw=2, alpha=0.85, label='ACV-PUQ-LASSO')
    ax.set_xlabel('Iteration k', fontsize=11)
    ax.set_ylabel('Objective F(x$^k$) [log]', fontsize=11)
    ax.set_title(f'Test {test_id}: {test_name}\nObjective Convergence (N=1000)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'PUQ: {obj_puq[-1]:.4e}', xy=(iters_puq[-1], obj_puq[-1]),
                fontsize=8, color='blue', ha='right')
    ax.annotate(f'ACV: {obj_acv[-1]:.4e}', xy=(iters_acv[-1], obj_acv[-1]),
                fontsize=8, color='red', ha='right')
    fig.tight_layout()
    fig.savefig(f'test{test_id}_objective.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    # ── Relative error ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(iters_puq, err_puq, 'b-', lw=2, alpha=0.85, label='PUQ-LASSO')
    ax.semilogy(iters_acv, err_acv, 'r--', lw=2, alpha=0.85, label='ACV-PUQ-LASSO')
    ax.set_xlabel('Iteration k', fontsize=11)
    ax.set_ylabel(r'$\|x^k - x^\star\| / \|x^\star\|$ [log]', fontsize=11)
    ax.set_title(f'Test {test_id}: {test_name}\nRelative Error (N=1000)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'PUQ: {err_puq[-1]:.6f}', xy=(iters_puq[-1], err_puq[-1]),
                fontsize=8, color='blue', ha='right')
    ax.annotate(f'ACV: {err_acv[-1]:.6f}', xy=(iters_acv[-1], err_acv[-1]),
                fontsize=8, color='red', ha='right')
    fig.tight_layout()
    fig.savefig(f'test{test_id}_relative_error.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    # ── Coefficients ──
    fig, ax = plt.subplots(figsize=(10, 4))
    n_c = len(x_true)
    idx = np.arange(n_c)
    ml, sl, bl = ax.stem(idx, x_true, linefmt='k-', markerfmt='ko',
                         basefmt='k-', label='True x*')
    plt.setp(sl, lw=0.8, alpha=0.5); plt.setp(ml, markersize=4)
    ax.scatter(idx, x_puq, c='blue', s=15, alpha=0.6, marker='x',
               linewidths=1.2, label='PUQ-LASSO', zorder=5)
    ax.scatter(idx, x_acv, c='red', s=15, alpha=0.6, marker='+',
               linewidths=1.2, label='ACV-PUQ-LASSO', zorder=5)
    ax.set_xlabel('Coefficient Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Test {test_id}: {test_name}\nRecovered Coefficients',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(f'test{test_id}_coefficients.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1–4: Sparse Recovery (N=1000)
# ═══════════════════════════════════════════════════════════════════════════════

def test_1():
    print("=" * 72)
    print("TEST 1: Ill-Conditioned Matrix, Low Noise (σ=0.1), N=1000")
    print("=" * 72)
    n, m, s, noise_std, N = 200, 80, 10, 0.1, 1000
    lam = adaptive_lambda(0.05, noise_std, n, m)
    A = build_ill_conditioned_matrix(m, n, cond_number=100.0)
    x_true = np.zeros(n)
    x_true[np.random.choice(n, s, replace=False)] = np.random.randn(s) * 3.0
    y = A @ x_true + noise_std * np.random.randn(m)
    x_puq, obj_p, err_p, it_p = run_puq_lasso(A, y, x_true, lam, N)
    x_acv, obj_a, err_a, it_a = run_acv_puq_lasso(A, y, x_true, lam, N)
    _print_results("1", x_puq, x_acv, x_true, err_p, err_a, obj_p, obj_a, A, N)
    save_test_figures(obj_p, obj_a, err_p, err_a, it_p, it_a,
                      x_puq, x_acv, x_true, "Ill-Cond, Low Noise", 1)
    return obj_p, obj_a, err_p, err_a, it_p, it_a, x_puq, x_acv, x_true, N


def test_2():
    print("\n" + "=" * 72)
    print("TEST 2: Ill-Conditioned, Moderate Noise (σ=0.5), N=1000")
    print("=" * 72)
    n, m, s, noise_std, N = 200, 80, 20, 0.5, 1000
    lam = adaptive_lambda(0.1, noise_std, n, m)
    A = build_ill_conditioned_matrix(m, n, cond_number=100.0)
    x_true = np.zeros(n)
    x_true[np.random.choice(n, s, replace=False)] = np.random.randn(s) * 2.0
    y = A @ x_true + noise_std * np.random.randn(m)
    x_puq, obj_p, err_p, it_p = run_puq_lasso(A, y, x_true, lam, N)
    x_acv, obj_a, err_a, it_a = run_acv_puq_lasso(A, y, x_true, lam, N)
    _print_results("2", x_puq, x_acv, x_true, err_p, err_a, obj_p, obj_a, A, N)
    save_test_figures(obj_p, obj_a, err_p, err_a, it_p, it_a,
                      x_puq, x_acv, x_true, "Ill-Cond, Moderate Noise", 2)
    return obj_p, obj_a, err_p, err_a, it_p, it_a, x_puq, x_acv, x_true, N


def test_3():
    print("\n" + "=" * 72)
    print("TEST 3: Correlated Dictionary (ρ=0.8), High Noise (σ=1.0), N=1000")
    print("=" * 72)
    n, m, s, noise_std, N = 150, 60, 8, 1.0, 1000
    lam = adaptive_lambda(0.2, noise_std, n, m)
    A = build_correlated_matrix(m, n, rho=0.8)
    x_true = np.zeros(n)
    x_true[np.random.choice(n, s, replace=False)] = np.random.randn(s) * 4.0
    y = A @ x_true + noise_std * np.random.randn(m)
    x_puq, obj_p, err_p, it_p = run_puq_lasso(A, y, x_true, lam, N)
    x_acv, obj_a, err_a, it_a = run_acv_puq_lasso(A, y, x_true, lam, N)
    _print_results("3", x_puq, x_acv, x_true, err_p, err_a, obj_p, obj_a, A, N)
    save_test_figures(obj_p, obj_a, err_p, err_a, it_p, it_a,
                      x_puq, x_acv, x_true, "Correlated, High Noise", 3)
    return obj_p, obj_a, err_p, err_a, it_p, it_a, x_puq, x_acv, x_true, N


def test_4():
    print("\n" + "=" * 72)
    print("TEST 4: Well-Conditioned, Noiseless, N=1000")
    print("=" * 72)
    n, m, s, noise_std, N = 100, 50, 5, 0.0, 1000
    lam = 0.01
    A = np.random.randn(m, n) / np.sqrt(m)
    x_true = np.zeros(n)
    sup = np.random.choice(n, s, replace=False)
    x_true[sup] = np.array([3.0, -2.5, 4.0, -1.5, 2.0])
    y = A @ x_true
    x_puq, obj_p, err_p, it_p = run_puq_lasso(A, y, x_true, lam, N)
    x_acv, obj_a, err_a, it_a = run_acv_puq_lasso(A, y, x_true, lam, N)
    _print_results("4", x_puq, x_acv, x_true, err_p, err_a, obj_p, obj_a, A, N)
    save_test_figures(obj_p, obj_a, err_p, err_a, it_p, it_a,
                      x_puq, x_acv, x_true, "Well-Cond, Noiseless", 4)
    return obj_p, obj_a, err_p, err_a, it_p, it_a, x_puq, x_acv, x_true, N


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Noise Sensitivity Sweep (N=1000)
# ═══════════════════════════════════════════════════════════════════════════════

def test_5():
    print("\n" + "=" * 72)
    print("TEST 5: Noise Sensitivity Sweep (N=1000, Adaptive λ)")
    print("=" * 72)
    n, m, s, N = 200, 80, 10, 1000
    lam_base = 0.05
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]

    A = build_ill_conditioned_matrix(m, n, cond_number=50.0)
    x_true = np.zeros(n)
    x_true[np.random.choice(n, s, replace=False)] = np.random.randn(s) * 3.0

    err_p_list, err_a_list = [], []
    obj_p_list, obj_a_list = [], []
    lam_list = []
    all_err_p, all_err_a, all_it = [], [], []

    for sig in noise_levels:
        lam = adaptive_lambda(lam_base, sig, n, m)
        lam_list.append(lam)
        y = A @ x_true + sig * np.random.randn(m)
        _, op, ep, itp = run_puq_lasso(A, y, x_true, lam, N)
        _, oa, ea, ita = run_acv_puq_lasso(A, y, x_true, lam, N)
        err_p_list.append(ep[-1]); err_a_list.append(ea[-1])
        obj_p_list.append(op[-1]); obj_a_list.append(oa[-1])
        all_err_p.append(ep); all_err_a.append(ea); all_it.append(itp)
        print(f"  σ={sig:.2f}, λ={lam:.4f} | PUQ err={ep[-1]:.6f} | ACV err={ea[-1]:.6f}")

    # ── 5A: Error bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    xp = np.arange(len(noise_levels)); w = 0.35
    ax.bar(xp - w/2, err_p_list, w, color='steelblue', alpha=0.8,
           label='PUQ-LASSO', edgecolor='black', lw=0.5)
    ax.bar(xp + w/2, err_a_list, w, color='indianred', alpha=0.8,
           label='ACV-PUQ-LASSO', edgecolor='black', lw=0.5)
    ax.set_xticks(xp)
    ax.set_xticklabels([f'σ={s:.2f}\nλ={l:.3f}' for s, l in zip(noise_levels, lam_list)], fontsize=7)
    ax.set_ylabel('Final Relative Error', fontsize=10)
    ax.set_title('Test 5A: Noise Sweep — Relative Error (N=1000)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('test5a_noise_sweep_error.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    # ── 5B: Objective bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xp - w/2, obj_p_list, w, color='steelblue', alpha=0.8,
           label='PUQ-LASSO', edgecolor='black', lw=0.5)
    ax.bar(xp + w/2, obj_a_list, w, color='indianred', alpha=0.8,
           label='ACV-PUQ-LASSO', edgecolor='black', lw=0.5)
    ax.set_xticks(xp)
    ax.set_xticklabels([f'σ={s:.2f}\nλ={l:.3f}' for s, l in zip(noise_levels, lam_list)], fontsize=7)
    ax.set_ylabel('Final Objective', fontsize=10)
    ax.set_title('Test 5B: Noise Sweep — Objective (N=1000)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('test5b_noise_sweep_objective.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    # ── 5C: Trajectories at selected noise levels ──
    sel = [0, 2, 4, 6]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for pi, si in enumerate(sel):
        ax = axes.flat[pi]
        ax.semilogy(all_it[si], all_err_p[si], 'b-', lw=1.5, label='PUQ-LASSO')
        ax.semilogy(all_it[si], all_err_a[si], 'r--', lw=1.5, label='ACV-PUQ-LASSO')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Rel Error [log]')
        ax.set_title(f'σ={noise_levels[si]:.2f}, λ={lam_list[si]:.4f}', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Test 5C: Error Trajectories (N=1000)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('test5c_noise_sweep_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    # ── 5D: Error monotonicity ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, err_a_list, 'r-o', lw=2, ms=8, label='ACV-PUQ-LASSO', zorder=5)
    ax.plot(noise_levels, err_p_list, 'b-s', lw=2, ms=7, alpha=0.7, label='PUQ-LASSO')
    ax.axhline(y=max(err_a_list), color='red', ls=':', alpha=0.5,
               label=f'ACV max err = {max(err_a_list):.6f}')
    ax.set_xlabel('Noise σ', fontsize=11); ax.set_ylabel('Final Rel Error', fontsize=11)
    ax.set_title('Test 5D: Error vs Noise (N=1000)\nACV error bounded', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('test5d_error_vs_noise.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    return noise_levels, err_p_list, err_a_list, obj_p_list, obj_a_list


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: IMAGE DENOISING — adrenal.png
# ═══════════════════════════════════════════════════════════════════════════════

def test_6_image_denoising():
    """
    Apply PUQ-LASSO and ACV-PUQ-LASSO to denoise 'adrenal.png'.
    
    Model: min_x  0.5‖x − y‖² + λ‖∇x‖₁
    where ∇ is the 2D discrete gradient (anisotropic total variation).
    
    Three noise levels are tested: σ = 0.05, 0.1, 0.2
    """
    print("\n" + "=" * 72)
    print("TEST 6: Image Denoising — adrenal.png (N=1000)")
    print("=" * 72)

    # ── Load image ──
    img_path = 'adrenal.png'
    if not os.path.exists(img_path):
        print(f"  [INFO] '{img_path}' not found. Creating synthetic medical-like image.")
        # Create a synthetic grayscale image resembling an adrenal scan
        np.random.seed(42)
        M, N_img = 128, 128
        clean = np.zeros((M, N_img))
        # Background
        clean[:] = 0.15
        # Organ-like elliptical regions
        yy, xx = np.ogrid[-64:64, -64:64]
        # Main organ
        mask1 = ((xx - 5) ** 2 / 30 ** 2 + (yy + 2) ** 2 / 25 ** 2) < 1
        clean[mask1] = 0.55
        # Lesion
        mask2 = ((xx + 10) ** 2 / 8 ** 2 + (yy - 5) ** 2 / 10 ** 2) < 1
        clean[mask2] = 0.80
        # Small bright spot
        mask3 = ((xx - 15) ** 2 + (yy + 15) ** 2) < 5 ** 2
        clean[mask3] = 0.95
        # Smooth edges
        from scipy.ndimage import gaussian_filter
        clean = gaussian_filter(clean, sigma=2.0)
        clean = np.clip(clean, 0, 1)
    else:
        clean = img_as_float(io.imread(img_path))
        if clean.ndim == 3:
            clean = color.rgb2gray(clean)
        # Resize if too large for speed
        if max(clean.shape) > 256:
            from skimage.transform import resize
            scale = 256 / max(clean.shape)
            new_shape = (int(clean.shape[0] * scale), int(clean.shape[1] * scale))
            clean = resize(clean, new_shape, anti_aliasing=True)
        clean = np.clip(clean, 0, 1)

    print(f"  Image size: {clean.shape}")

    noise_levels_img = [0.05, 0.10, 0.20]
    N_iter = 1000
    record_every = 20

    for sigma_n in noise_levels_img:
        print(f"\n  --- Noise σ = {sigma_n} ---")

        np.random.seed(0)
        noisy = clean + sigma_n * np.random.randn(*clean.shape)
        noisy = np.clip(noisy, 0, 1)

        # Adaptive λ for TV denoising
        lam = 0.02 + 0.3 * sigma_n

        psnr_noisy = psnr(clean, noisy)
        ssim_noisy = ssim_simple(clean, noisy)
        print(f"  Noisy PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")

        # ── PUQ denoising ──
        den_puq, psnr_puq, ssim_puq, it_puq = run_puq_image_denoise(
            noisy, clean, lam, N_iter, record_every)

        # ── ACV denoising ──
        den_acv, psnr_acv, ssim_acv, it_acv = run_acv_image_denoise(
            noisy, clean, lam, N_iter, record_every)

        psnr_puq_final = psnr(clean, den_puq)
        psnr_acv_final = psnr(clean, den_acv)
        ssim_puq_final = ssim_simple(clean, den_puq)
        ssim_acv_final = ssim_simple(clean, den_acv)

        print(f"  PUQ-LASSO  — PSNR: {psnr_puq_final:.2f} dB, SSIM: {ssim_puq_final:.4f}")
        print(f"  ACV-PUQ    — PSNR: {psnr_acv_final:.2f} dB, SSIM: {ssim_acv_final:.4f}")
        print(f"  Δ PSNR: {psnr_acv_final - psnr_puq_final:+.2f} dB, "
              f"Δ SSIM: {ssim_acv_final - ssim_puq_final:+.4f}")

        sig_str = f'{sigma_n:.2f}'.replace('.', '')

        # ── Figure 6A: Image comparison ──
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original', fontsize=10, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Noisy (σ={sigma_n})\nPSNR={psnr_noisy:.1f}dB',
                          fontsize=10, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(den_puq, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'PUQ-LASSO\nPSNR={psnr_puq_final:.1f}dB\n'
                          f'SSIM={ssim_puq_final:.3f}',
                          fontsize=10, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(den_acv, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f'ACV-PUQ-LASSO\nPSNR={psnr_acv_final:.1f}dB\n'
                          f'SSIM={ssim_acv_final:.3f}',
                          fontsize=10, fontweight='bold')
        axes[3].axis('off')

        fig.suptitle(f'Test 6: Image Denoising (σ={sigma_n}, λ={lam:.3f}, N={N_iter})',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f'test6a_images_sigma{sig_str}.png', dpi=150, bbox_inches='tight')
        plt.show(); plt.close(fig)

        # ── Figure 6B: PSNR convergence ──
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(it_puq, psnr_puq, 'b-', lw=2, alpha=0.85, label='PUQ-LASSO')
        ax.plot(it_acv, psnr_acv, 'r--', lw=2, alpha=0.85, label='ACV-PUQ-LASSO')
        ax.axhline(y=psnr_noisy, color='gray', ls=':', alpha=0.7, label=f'Noisy ({psnr_noisy:.1f}dB)')
        ax.set_xlabel('Iteration k', fontsize=11)
        ax.set_ylabel('PSNR (dB)', fontsize=11)
        ax.set_title(f'Test 6B: PSNR Convergence (σ={sigma_n}, N={N_iter})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'test6b_psnr_sigma{sig_str}.png', dpi=150, bbox_inches='tight')
        plt.show(); plt.close(fig)

        # ── Figure 6C: SSIM convergence ──
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(it_puq, ssim_puq, 'b-', lw=2, alpha=0.85, label='PUQ-LASSO')
        ax.plot(it_acv, ssim_acv, 'r--', lw=2, alpha=0.85, label='ACV-PUQ-LASSO')
        ax.axhline(y=ssim_noisy, color='gray', ls=':', alpha=0.7, label=f'Noisy ({ssim_noisy:.3f})')
        ax.set_xlabel('Iteration k', fontsize=11)
        ax.set_ylabel('SSIM', fontsize=11)
        ax.set_title(f'Test 6C: SSIM Convergence (σ={sigma_n}, N={N_iter})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'test6c_ssim_sigma{sig_str}.png', dpi=150, bbox_inches='tight')
        plt.show(); plt.close(fig)

        # ── Figure 6D: Residual images ──
        res_puq = np.abs(clean - den_puq)
        res_acv = np.abs(clean - den_acv)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        vmax_res = max(res_puq.max(), res_acv.max())

        axes[0].imshow(np.abs(clean - noisy), cmap='hot', vmin=0, vmax=vmax_res)
        axes[0].set_title(f'|Original − Noisy|\nMean={np.mean(np.abs(clean-noisy)):.4f}',
                          fontsize=10, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(res_puq, cmap='hot', vmin=0, vmax=vmax_res)
        axes[1].set_title(f'|Original − PUQ|\nMean={np.mean(res_puq):.4f}',
                          fontsize=10, fontweight='bold')
        axes[1].axis('off')

        im = axes[2].imshow(res_acv, cmap='hot', vmin=0, vmax=vmax_res)
        axes[2].set_title(f'|Original − ACV|\nMean={np.mean(res_acv):.4f}',
                          fontsize=10, fontweight='bold')
        axes[2].axis('off')

        fig.colorbar(im, ax=axes, shrink=0.8, label='Absolute Error')
        fig.suptitle(f'Test 6D: Residual Maps (σ={sigma_n})',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 0.92, 0.93])
        fig.savefig(f'test6d_residuals_sigma{sig_str}.png', dpi=150, bbox_inches='tight')
        plt.show(); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SPEEDUP RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speedup_ratio(results, labels):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    for i, r in enumerate(results):
        obj_p, obj_a, _, _, it_p, it_a = r[0], r[1], r[2], r[3], r[4], r[5]
        # Use common iterations
        min_len = min(len(obj_p), len(obj_a))
        ratio = obj_a[:min_len] / (obj_p[:min_len] + 1e-15)
        ax.plot(it_p[:min_len], ratio, color=colors[i], lw=1.5, label=labels[i])

    ax.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('Iteration k', fontsize=11)
    ax.set_ylabel('Ratio F$_{ACV}$/F$_{PUQ}$', fontsize=11)
    ax.set_title('Speedup Ratio (N=1000)\nValues < 1 ⟹ ACV faster',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.5])
    fig.tight_layout()
    fig.savefig('speedup_ratio_all_tests.png', dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Sparse recovery tests (N=1000)
    r1 = test_1()
    r2 = test_2()
    r3 = test_3()
    r4 = test_4()
    r5 = test_5()

    # Image denoising test
    test_6_image_denoising()

    # Speedup ratio
    labels = ['Test 1: Ill-Cond σ=0.1', 'Test 2: Ill-Cond σ=0.5',
              'Test 3: Correlated σ=1.0', 'Test 4: Well-Cond σ=0']
    plot_speedup_ratio([r1, r2, r3, r4], labels)

    # ── Summary ──
    print("\n" + "=" * 72)
    print("SAVED FILES SUMMARY")
    print("=" * 72)
    files = [
        # Tests 1-4
        *[(f'test{i}_{s}.png', f'Test {i} {s}')
          for i in range(1, 5)
          for s in ['objective', 'relative_error', 'coefficients']],
        # Test 5
        ('test5a_noise_sweep_error.png', 'Noise sweep error bars'),
        ('test5b_noise_sweep_objective.png', 'Noise sweep objective bars'),
        ('test5c_noise_sweep_trajectories.png', 'Error trajectories at 4 σ'),
        ('test5d_error_vs_noise.png', 'Error monotonicity check'),
        # Test 6
        *[(f'test6a_images_sigma{s}.png', f'Denoised images σ={v}')
          for s, v in [('005', 0.05), ('010', 0.10), ('020', 0.20)]],
        *[(f'test6b_psnr_sigma{s}.png', f'PSNR convergence σ={v}')
          for s, v in [('005', 0.05), ('010', 0.10), ('020', 0.20)]],
        *[(f'test6c_ssim_sigma{s}.png', f'SSIM convergence σ={v}')
          for s, v in [('005', 0.05), ('010', 0.10), ('020', 0.20)]],
        *[(f'test6d_residuals_sigma{s}.png', f'Residual maps σ={v}')
          for s, v in [('005', 0.05), ('010', 0.10), ('020', 0.20)]],
        # Speedup
        ('speedup_ratio_all_tests.png', 'Speedup ratio Tests 1-4'),
    ]
    for fname, desc in files:
        print(f"  {fname:45s} — {desc}")
    print(f"\n  Total: {len(files)} figures saved.")
    print("=" * 72)