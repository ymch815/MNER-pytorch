# MNE PyTorch Implementation

A PyTorch-based implementation of the Low-Rank Maximum Noise Entropy (MNER) model. 

## Context

This PyTorch implementation is a modernized version of the original Low-rank Maximum Noise Entropy (MNER) algorithm, migrated from Theano to PyTorch for improved performance and maintainability.

**Original MNEr Resources:**
- **Tutorial**: [MNER Tutorial](http://joelkaardal.com/links/tutorials/mner.html) - Comprehensive introduction to the MNE method
- **Original Code**: [mner](https://github.com/jkaardal/mner) - Theano-based reference implementation

### Key Features of This Implementation

This lightweight version focuses on the core MNEr functionality with the following optimized configuration:

- **Training Strategy**: `demo_type = 1` - Optimized for local minimum convergence on training data
- **Constraint System**: `cetype = ["UV-linear-insert"]` - UV-factorization with linear insertion constraints
- **Optimization Backend**: `LBFGSSolver` - L-BFGS solver with automatic differentiation

### Improvements Over Original

- **PyTorch Integration**: Leverages automatic differentiation for exact gradients
- **Numerical Stability**: Uses stable loss functions and optimized convergence criteria
- **Modern Dependencies**: Built on current scientific Python stack (PyTorch, scikit-learn)
- **Enhanced Testing**: Comprehensive validation with parameter recovery analysis

### Future Development

Future updates may extend this implementation to include additional features from the original Theano-based MNEr repository, such as alternative solvers and constraint types.

## Mathematical Foundation

The MNE model predicts the probability of neural response as:

```
P(y=1|s) = σ(a + h^T s + s^T J s)
```

Where:
- `σ` is the sigmoid function
- `a` is the bias term
- `h` is the linear weight vector
- `J = UV^T` is the interaction matrix (low-rank factorized)
- `s` is the stimulus feature vector

The low-rank factorization `J = UV^T` dramatically reduces parameters from `O(d²)` to `O(d·r)` where `d` is feature dimension and `r` is rank.

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0 (for testing/visualization)

## Testing

Run the test suite to verify installation:

```bash
# Run test
python tests/test_synthetic_pytorch.py
```
