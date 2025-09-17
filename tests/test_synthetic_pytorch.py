"""
PyTorch MNE Test Suite with Synthetic Data
==========================================

This test follows the same synthetic data generation and visualization flow as the original 
Theano-based tests, but uses the PyTorch implementation for optimization.

The test generates correlated Gaussian noise images with synthetic weights and validates
the PyTorch MNE model's ability to recover ground truth parameters.
"""

from __future__ import print_function

import os
import sys
import numpy as np
import torch


np.random.seed(42)
# Add parent directory to path for imports
here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path = [here] + sys.path

import mner.optimizer
import mner.util.util

import matplotlib.pyplot as plt
matplotlib_loaded = True

# Configuration
float_dtype = 'float64'  # String for PyTorch compatibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test parameters
demo_type = 1
show_plot = True

# Data split parameters
train_fraction = 0.7
cv_fraction = 0.2

# Generate correlated Gaussian noise images (same as original)
ny = 16
nx = 16  
nsamp = 100000
ndim = ny * nx

print(f"Generating synthetic data: {nsamp} samples x {ndim} features ({ny}x{nx} images)")

def GP(x, y, x0, y0):
    """Gaussian Process covariance function"""
    Kinv = 1.0 / 3.0
    r = np.array([x - x0, y - y0]).reshape((2, 1))
    return np.exp(-np.dot(r.T, r) * Kinv)

# Build covariance matrix
print("Building spatial covariance matrix...")
C = np.zeros((ndim, ndim))
for i in range(ny):
    for j in range(nx):
        c = np.zeros((ny, nx))
        for k in range(ny):
            for l in range(nx):
                c[l, k] = GP(l, k, j, i)
        C[:, i * nx + j] = c.ravel()

C = np.dot(C.T, C)

# Generate correlated features using SVD
L, M, R = np.linalg.svd(C)
X = np.dot(np.diag(np.sqrt(M)), R)
s = np.dot(np.random.randn(nsamp, ndim), X)

# Feature scaling - TEMPORARILY DISABLED to test hypothesis
# print("Skipping z-score normalization to test J recovery...")
# s, s_avg, s_std = mner.util.util.zscore_features(s)  # Disabled

# Generate synthetic weights using 2D Gaussians
def Gauss2D(xi, x0, sx, yi, y0, sy, A, phi):
    """Generate 2D Gaussian pattern"""
    xm = np.tile(xi.reshape((1, xi.size)), (yi.size, 1))
    ym = np.tile(yi.reshape((yi.size, 1)), (1, xi.size))

    x = (xm - x0) * np.cos(phi) + (ym - y0) * np.sin(phi)
    y = (x0 - xm) * np.sin(phi) + (ym - y0) * np.cos(phi)

    return A * np.exp(-((x / sx) ** 2 + (y / sy) ** 2) / 2.0)

# Create spatial grid
xi = np.arange(1.0, 17.0, 1.0)
yi = np.arange(1.0, 17.0, 1.0)

# Generate basis functions F (same as original)
print("Creating synthetic interaction basis functions...")
F = np.zeros((ndim, 6))
F[:, 0] = np.reshape(Gauss2D(xi, 8.0, 1.5, yi, 11.0, 1.5, 1.0, 0.0), (ndim,))
F[:, 1] = np.reshape(Gauss2D(xi, 11.0, 1.5, yi, 11.0, 1.5, -0.5, 0.0), (ndim,))
F[:, 2] = np.reshape(Gauss2D(xi, 5.0, 1.5, yi, 11.0, 1.5, -0.5, 0.0), (ndim,))
F[:, 3] = np.reshape(Gauss2D(xi, 8.0, 1.5, yi, 8.0, 2.0, -0.5, 0.0), (ndim,))
F[:, 4] = np.reshape(Gauss2D(xi, 5.0, 1.5, yi, 8.0, 2.0, 0.3, -np.pi/4.0), (ndim,))
F[:, 5] = np.reshape(Gauss2D(xi, 11.0, 1.5, yi, 8.0, 2.0, 0.3, np.pi/4.0), (ndim,))

# Generate linear coupling weights h (same as original)
h_GT = np.reshape(Gauss2D(xi, 8.0, 1.0, yi, 11.0, 1.0, 0.1, 0.0), (ndim, 1))

print(f"Generated {F.shape[1]} basis functions and linear coupling weights")

# Generate responses using MNE model (same as original)
print("Generating synthetic neural responses...")
a_GT = -3.5  # Global bias
g = 0.05     # Interaction strength
W = np.diag(np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0]))  # Eigenvalue signs
J_GT = g * np.dot(F, np.dot(W, F.T))  # Ground truth interaction matrix

# Compute logits: a + h^T*s + s^T*J*s
x_logits = a_GT + np.dot(s, h_GT).ravel() + np.sum(s * np.dot(s, J_GT), axis=1)
p_true = 1.0 / (1.0 + np.exp(-x_logits))

# Generate binary responses
y = np.zeros((nsamp,))
rnd = np.random.rand(nsamp,)
y[rnd < p_true] = 1.0

print(f"Response probability: {np.mean(y):.3f}")

# Generate dataset splits
print("Splitting into train/validation/test sets...")
trainset, cvset, testset, _ = mner.util.util.generate_dataset_logical_indices(
    train_fraction, cv_fraction, nsamp
)

datasets = {'trainset': trainset, 'cvset': cvset, 'testset': testset}
print(f"  - Training samples: {np.sum(trainset)}")
print(f"  - Validation samples: {np.sum(cvset)}")
print(f"  - Test samples: {np.sum(testset)}")

# Model parameters - UPDATED with implementation parity fixes
rank = 6
cetype = ["UV-linear-insert"]
rtype = ["nuclear-norm", "l2-norm"]  # UPDATED: Use dual regularization
# rtype = ["nuclear-norm"]
# Set signs of eigenvalues for symmetrization
# csigns = np.array([-1, 1] * int(rank / 2))
csigns = np.array([1,-1,-1,-1,1,1])

# Set scaling of cost function (for PyTorch version, use scalar)
# Note: PyTorch version computes negative log-likelihood for minimization, so use fscale = 1
fscale = 1

# PyTorch optimization parameters - optimized for synthetic data
nuclear_norm_param = 0.03  # For UV interaction regularization
l2_norm_param = 0.03        # For h linear regularization
max_iter = 3000             # Reduced from 10000 - usually sufficient
ftol = 1e-9                 # Tighter function tolerance for better recovery, -6 to -9
gtol = 1e-9                 # Tighter gradient tolerance for better recovery, -6 to -8

print(f"\nSetting up PyTorch MNE optimizer...")
print(f"  - Model rank: {rank}")
print(f"  - Constraint: {cetype[0]}")
print(f"  - Regularization: {rtype} (λ_nuclear={nuclear_norm_param}, λ_l2={l2_norm_param})")
print(f"  - Device: {device}")

# Set up the PyTorch optimizer - UPDATED with dual regularization parameters
opt = mner.optimizer.Optimizer(
    y, s, rank,
    cetype=cetype,
    rtype=rtype, 
    datasets=datasets,
    fscale=fscale,
    csigns=csigns,
    nuclear_norm_param=nuclear_norm_param,
    l2_norm_param=l2_norm_param,
    maxiter=max_iter,
    ftol=ftol,              # Function tolerance
    gtol=gtol,              # Gradient tolerance  
    device=device,
    float_dtype=float_dtype,
    verbosity=2
)

# Optimize the model
print(f"\nRunning PyTorch optimization...")
# Use Theano-compatible initialization for better parameter recovery
results = opt.optimize(init_method='theano_compatible', seed=42)

# Extract results
x_opt = results['xopt']
f_train = results['fopt']
f_cv = results.get('cv_cost', 'N/A')
f_test = results.get('test_cost', 'N/A')

print(f'\nOptimization Results:')
print(f'  - Final training cost: {f_train:.6f}')
print(f'  - Final CV cost: {f_cv}')
print(f'  - Final test cost: {f_test}')

# Convert learned parameters back to weight matrices
model = results['model']
model.enable_constraint_verification(True)  # Enable constraint verification
x_torch = torch.tensor(x_opt, dtype=torch.float64, device=device)
a_learned, h_learned, U_learned, V_learned = model.vec_to_weights(x_torch)

# Convert to numpy for analysis
if a_learned is not None:
    a_learned = a_learned.cpu().numpy()
    print(f'  - Learned bias: {a_learned[0]:.4f} (true: {a_GT:.4f})')

if h_learned is not None:
    if hasattr(h_learned, 'cpu'):
        h_learned = h_learned.cpu().numpy()
    else:
        h_learned = np.array(h_learned)
    
    # Compute correlation with ground truth
    h_corr = np.corrcoef(h_GT.flatten(), h_learned.flatten())[0, 1]
    print(f'  - Linear weights correlation: {h_corr:.4f}')

if U_learned is not None and V_learned is not None:
    if hasattr(U_learned, 'cpu'):
        U_learned = U_learned.cpu().numpy()
        V_learned = V_learned.cpu().numpy()
    
    # Apply constraint signs and form interaction matrix
    V_constrained = U_learned * csigns[np.newaxis, :]
    J_learned = np.dot(U_learned, V_constrained.T)
    
    # Symmetrize
    J_learned_sym = 0.5 * (J_learned + J_learned.T)
    J_GT_sym = 0.5 * (J_GT + J_GT.T)
    
    # Compute correlation
    J_corr = np.corrcoef(J_GT_sym.flatten(), J_learned_sym.flatten())[0, 1]
    print(f'  - Interaction matrix correlation: {J_corr:.4f}')
    
    # Compute SVD for eigenvector comparison
    u_learned, _, _ = np.linalg.svd(J_learned_sym)
    u_GT, _, _ = np.linalg.svd(J_GT_sym)
    
    print(f'  - Learned U shape: {U_learned.shape}')
    print(f'  - Learned interaction matrix rank: {np.linalg.matrix_rank(J_learned_sym)}')

# Generate comprehensive visualization
if matplotlib_loaded and show_plot:
    print(f"\nGenerating visualization plots...")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('PyTorch MNE: Ground Truth vs Learned Parameters', fontsize=16, fontweight='bold')
    
    # 1. Linear coupling weights comparison
    ax1 = plt.subplot(3, 4, 1)
    cm = np.max(np.abs(h_GT))
    plt.imshow(np.reshape(h_GT, (ny, nx)), aspect='equal', interpolation='none', 
               clim=(-cm, cm), cmap='RdBu')
    plt.title('Ground Truth h', fontsize=12)
    plt.colorbar(shrink=0.8)
    
    ax2 = plt.subplot(3, 4, 2)
    if h_learned is not None:
        cm = np.max(np.abs(h_learned))
        plt.imshow(np.reshape(h_learned, (ny, nx)), aspect='equal', interpolation='none', 
                   clim=(-cm, cm), cmap='RdBu')
        plt.title(f'Learned h (r={h_corr:.3f})', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No h learned', ha='center', va='center', transform=ax2.transAxes)
        plt.title('Learned h', fontsize=12)
    plt.colorbar(shrink=0.8)
    
    # 2. Linear weights correlation scatter
    ax3 = plt.subplot(3, 4, 3)
    if h_learned is not None:
        plt.scatter(h_GT.flatten(), h_learned.flatten(), alpha=0.6, s=20)
        plt.plot([h_GT.min(), h_GT.max()], [h_GT.min(), h_GT.max()], 'r--', alpha=0.8)
        plt.xlabel('True h')
        plt.ylabel('Learned h')
        plt.title(f'Linear Weights\n(r = {h_corr:.3f})')
        plt.grid(True, alpha=0.3)
    
    # 3. Bias comparison
    ax4 = plt.subplot(3, 4, 4)
    bias_learned = a_learned[0] if a_learned is not None else 0
    plt.bar(['True', 'Learned'], [a_GT, bias_learned], color=['blue', 'red'], alpha=0.7)
    plt.title('Bias Comparison')
    plt.ylabel('Bias Value')
    plt.grid(True, alpha=0.3)
    
    # 4. Ground truth eigenvectors (top row)
    for i in range(6):
        ax = plt.subplot(3, 6, 6 + i + 1)
        if i < u_GT.shape[1]:
            cm = np.max(np.abs(u_GT[:, i]))
            plt.imshow(np.reshape(u_GT[:, i], (ny, nx)), aspect='equal', 
                      interpolation='none', clim=(-cm, cm), cmap='RdBu')
            plt.title(f'GT EV {i+1}', fontsize=10)
        plt.axis('off')
    
    # 5. Learned eigenvectors (bottom row)  
    for i in range(6):
        ax = plt.subplot(3, 6, 12 + i + 1)
        if 'u_learned' in locals() and i < u_learned.shape[1]:
            cm = np.max(np.abs(u_learned[:, i]))
            plt.imshow(np.reshape(u_learned[:, i], (ny, nx)), aspect='equal',
                      interpolation='none', clim=(-cm, cm), cmap='RdBu')
            plt.title(f'Learn EV {i+1}', fontsize=10)
        else:
            plt.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            plt.title(f'Learn EV {i+1}', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'pytorch_synthetic_test_results.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"  - Saved visualization: {plot_filename}")
    
    # # Show the plot
    # plt.show()
    
    # Additional detailed correlation plots
    if h_learned is not None and 'J_learned_sym' in locals():
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig2.suptitle('Detailed Parameter Recovery Analysis', fontsize=14, fontweight='bold')
        
        # Linear weights detailed comparison
        axes[0, 0].scatter(h_GT.flatten(), h_learned.flatten(), alpha=0.6, s=30)
        axes[0, 0].plot([h_GT.min(), h_GT.max()], [h_GT.min(), h_GT.max()], 'r--')
        axes[0, 0].set_xlabel('Ground Truth Linear Weights')
        axes[0, 0].set_ylabel('Learned Linear Weights')
        axes[0, 0].set_title(f'Linear Coupling Recovery (r = {h_corr:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Interaction matrix comparison
        axes[0, 1].scatter(J_GT_sym.flatten(), J_learned_sym.flatten(), alpha=0.4, s=10)
        axes[0, 1].plot([J_GT_sym.min(), J_GT_sym.max()], [J_GT_sym.min(), J_GT_sym.max()], 'r--')
        axes[0, 1].set_xlabel('Ground Truth Interaction Matrix')
        axes[0, 1].set_ylabel('Learned Interaction Matrix')
        axes[0, 1].set_title(f'Interaction Matrix Recovery (r = {J_corr:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Eigenvalue comparison
        gt_eigenvals = np.linalg.eigvals(J_GT_sym)
        learned_eigenvals = np.linalg.eigvals(J_learned_sym)
        gt_eigenvals = np.sort(gt_eigenvals)[::-1]  # Sort descending
        learned_eigenvals = np.sort(learned_eigenvals)[::-1]
        
        axes[1, 0].plot(range(len(gt_eigenvals)), gt_eigenvals, 'bo-', label='Ground Truth', alpha=0.7)
        axes[1, 0].plot(range(len(learned_eigenvals)), learned_eigenvals, 'ro-', label='Learned', alpha=0.7)
        axes[1, 0].set_xlabel('Eigenvalue Index')
        axes[1, 0].set_ylabel('Eigenvalue')
        axes[1, 0].set_title('Eigenvalue Spectrum Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training progress (if available)
        opt_info = results.get('optimization_info', {})
        if isinstance(opt_info, dict) and 'loss_history' in opt_info:
            axes[1, 1].plot(opt_info['loss_history'], 'b-', linewidth=2)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Progress')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Training history\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Training Progress')
        
        plt.tight_layout()
        
        # Save detailed analysis
        detail_filename = 'pytorch_synthetic_detailed_analysis.png'
        plt.savefig(detail_filename, dpi=300, bbox_inches='tight')
        print(f"  - Saved detailed analysis: {detail_filename}")
        
        # plt.show()

# Summary
print(f"\n" + "="*60)
print("PyTorch MNE Synthetic Data Test Completed! ✅")
print("="*60)
print(f"Key Results:")
if h_learned is not None:
    print(f"  - Linear coupling recovery: r = {h_corr:.3f}")
if 'J_corr' in locals():
    print(f"  - Interaction matrix recovery: r = {J_corr:.3f}")
print(f"  - Model rank: {rank} (true rank: {np.linalg.matrix_rank(J_GT_sym)})")