from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import svd

""" model.py (module)

    This module contains the PyTorch implementation of the MNEr model
    for low-rank Maximum Noise Entropy analysis with automatic differentiation.

"""


class MNEr(nn.Module):
    """ MNEr (class)

        PyTorch implementation of the low-rank MNE model with automatic
        differentiation, supporting UV-linear-insert constraints and
        nuclear-norm regularization.

    """

    def __init__(self, resp, feat, rank, cetype=None, citype=None, rtype=None, fscale=1.0, 
                 use_vars=None, use_consts=None, **kwargs):
        """Initialize MNEr class instantiation.

            [inputs] (resp, feat, rank, cetype=None, citype=None,
              rtype=None, fscale=1.0, use_vars=None, use_consts=None, **kwargs)
                resp: numpy array of the output labels with shape
                  (nsamp,) where nsamp is the number of data
                  samples. Each element of resp must be in the range
                  [0, 1].
                feat: numpy array of the input features with shape
                  (nsamp, ndim) where ndim is the number of features.
                rank: positive integer that sets the number of columns
                  of the matrices U and V (that both have shape (ndim,
                  rank)).
                cetype: (optional) list of strings that tell the class
                  which equality constraints are being used. 
                  Supported: ["UV-linear-insert"]
                citype: (optional) inequality constraints (not supported)
                rtype: (optional) list of regularization penalties.
                  Supported: ["nuclear-norm"]
                fscale: (optional) scaling factor for the objective function
                use_vars: (optional) dictionary of variables to use
                use_consts: (optional) dictionary of constants to use
                kwargs: keyword arguments including:
                  - device: 'cpu' or 'cuda' 
                  - float_dtype: 'float32' or 'float64'
                  - csigns: constraint signs for UV-linear-insert

        """
        super(MNEr, self).__init__()

        # Device and dtype management
        self.device = kwargs.get('device', 'cpu')
        self.float_dtype_str = kwargs.get('float_dtype', 'float64')
        self.float_dtype = getattr(torch, self.float_dtype_str)

        # Get number of samples and features
        self.nsamp = resp.size if isinstance(resp, np.ndarray) else len(resp)
        self.ndim = feat.shape[1]
        assert self.nsamp == feat.shape[0], "Response and feature arrays must have same number of samples"

        # Convert data to tensors and move to device
        self.resp = torch.tensor(resp.reshape(-1), dtype=self.float_dtype, device=self.device)
        self.feat = torch.tensor(feat, dtype=self.float_dtype, device=self.device)
        
        # Model parameters
        self.rank = rank
        self.fscale = fscale

        # Constraint and regularization settings
        self.cetype = cetype or []
        if isinstance(self.cetype, str):
            self.cetype = [self.cetype]
            
        self.citype = citype or []
        if isinstance(self.citype, str):
            self.citype = [self.citype]
            
        self.rtype = rtype or []
        if isinstance(self.rtype, str):
            self.rtype = [self.rtype]

        # Variable usage settings
        if use_vars is None:
            use_vars = {'avar': True, 'hvar': True, 'UVvar': True}
        self.use_vars = use_vars
        
        if use_consts is None:
            use_consts = {'aconst': False, 'hconst': False, 'UVconst': False, 'Jconst': False}
        self.use_consts = use_consts

        # Initialize constraint signs if needed
        if "UV-linear-insert" in self.cetype:
            csigns = kwargs.get('csigns', np.ones(rank))
            self.csigns = torch.tensor(csigns, dtype=self.float_dtype, device=self.device)
        else:
            self.csigns = None

        # Initialize regularization parameters
        self.reg_params = {}
        if "nuclear-norm" in self.rtype:
            reg_param_value = kwargs.get('nuclear_norm_param', 0.01)
            self.reg_params['nuclear-norm'] = torch.tensor(reg_param_value, dtype=self.float_dtype, device=self.device)
        if "l2-norm" in self.rtype:
            l2_param_value = kwargs.get('l2_norm_param', 0.01)
            self.reg_params['l2-norm'] = torch.tensor(l2_param_value, dtype=self.float_dtype, device=self.device)

        # Initialize constraint verification (disabled by default for performance)
        self._verify_constraints_enabled = False
        
        # Initialize parameter size calculation
        self._compute_param_size()

    def _compute_param_size(self):
        """Compute the total number of parameters"""
        size = 0
        
        # Intercept parameter
        if self.use_vars['avar']:
            size += 1
            
        # Linear parameters
        if self.use_vars['hvar']:
            size += self.ndim
            
        # Low-rank parameters
        if self.use_vars['UVvar']:
            if "UV-linear-insert" in self.cetype:
                # Only need U parameters, V is determined by constraint
                size += self.ndim * self.rank
            else:
                # Need both U and V parameters
                size += 2 * self.ndim * self.rank
                
        self.param_size = size

    def get_param_size(self):
        """Return the total number of parameters"""
        return self.param_size

    def vec_to_weights(self, x):
        """Convert parameter vector to weight matrices
        
        Args:
            x: torch tensor of shape (param_size,) containing parameters
            
        Returns:
            tuple: (a, h, U, V) weight matrices
        """
        
        if x.numel() == 1:
            # Only intercept
            a = x[:1]
            h, U, V = None, None, None
            
        elif x.numel() == (1 + self.ndim):
            # Intercept + linear term
            a = x[:1]
            h = x[1:self.ndim+1]
            U, V = None, None
            
        elif x.numel() == (1 + self.ndim + self.ndim*self.rank):
            # Intercept + linear + single low-rank matrix (UV-linear-insert)
            a = x[:1]
            h = x[1:self.ndim+1]
            U = x[1+self.ndim:].reshape(self.ndim, self.rank)
            
            # Apply UV-linear-insert constraint
            if "UV-linear-insert" in self.cetype:
                V = U * self.csigns.unsqueeze(0)  # Broadcasting
            else:
                V = None
                
        elif x.numel() == (1 + self.ndim + 2*self.ndim*self.rank):
            # Intercept + linear + both U and V matrices
            a = x[:1]
            h = x[1:self.ndim+1]
            Q = x[1+self.ndim:].reshape(2*self.ndim, self.rank)
            U = Q[:self.ndim, :]
            V = Q[self.ndim:, :]
            
        else:
            # Empty case
            a = None
            h = None 
            U = None
            V = None

        # Verify constraints are satisfied
        if self._verify_constraints_enabled:
            self._verify_constraints(a, h, U, V)
            
        return a, h, U, V

    def _verify_constraints(self, a, h, U, V):
        """Verify that constraints are satisfied"""
        if "UV-linear-insert" in self.cetype and U is not None and V is not None:
            # Check UV-linear-insert constraint: V[:, k] = csigns[k] * U[:, k]
            expected_V = U * self.csigns.unsqueeze(0)
            constraint_error = torch.norm(V - expected_V)
            if constraint_error > 1e-10:
                print(f"WARNING: UV-linear-insert constraint violation: error = {constraint_error.item():.2e}")

    def enable_constraint_verification(self, enabled=True):
        """Enable or disable constraint verification"""
        self._verify_constraints_enabled = enabled
    
    def project_to_constraints(self, x):
        """Project parameter vector onto constraint manifold
        
        Args:
            x: parameter vector tensor
            
        Returns:
            torch tensor: projected parameter vector that satisfies constraints
        """
        # For UV-linear-insert constraint, this is automatically satisfied in vec_to_weights
        # by construction, so just return x
        return x

    def weights_to_vec(self, a, h, U, V=None, **kwargs):
        """Convert weight matrices to parameter vector
        
        Args:
            a: intercept parameter tensor of shape (1,)
            h: linear parameter tensor of shape (ndim,) or None  
            U: first low-rank matrix tensor of shape (ndim, rank) or None
            V: second low-rank matrix tensor of shape (ndim, rank) or None
            
        Returns:
            torch tensor: flattened parameter vector
        """
        
        if a is None:
            return torch.tensor([], dtype=self.float_dtype, device=self.device)
        
        assert a.numel() == 1, "Intercept must be scalar"
        
        components = [a.reshape(1)]
        
        if h is not None:
            components.append(h.reshape(-1))
            
        if U is not None:
            if "UV-linear-insert" in self.cetype:
                # Only store U, V is determined by constraint
                components.append(U.reshape(-1))
            elif V is not None:
                # Store both U and V
                Q = torch.cat([U, V], dim=0)
                components.append(Q.reshape(-1))
            else:
                components.append(U.reshape(-1))
        
        return torch.cat(components)

    def compute_cost(self, x):
        """Compute the negative log-likelihood cost function with regularization
        
        Args:
            x: parameter vector tensor of shape (param_size,)
            
        Returns:
            torch tensor: scalar cost value
        """
        
        # Convert parameters to weight matrices
        a, h, U, V = self.vec_to_weights(x)
        
        # Initialize the argument of the logistic function
        # P(y=1|s) = 1 / (1 + exp(-arg)) where arg = a + h^T s + s^T J s
        if a is not None:
            arg = a.expand(self.nsamp)  # Broadcast to all samples
        else:
            arg = torch.zeros(self.nsamp, dtype=self.float_dtype, device=self.device)
        
        # Add linear term: h^T s
        if h is not None:
            linear_term = torch.matmul(self.feat, h)
            arg = arg + linear_term
        
        # Add quadratic term: s^T J s where J = U V^T
        if U is not None and V is not None:
            # Compute (U^T s) and (V^T s)
            Us = torch.matmul(self.feat, U)  # (nsamp, rank)
            Vs = torch.matmul(self.feat, V)  # (nsamp, rank)
            
            # Quadratic term: sum over rank of (U^T s) * (V^T s) 
            quad_term = torch.sum(Us * Vs, dim=1)  # (nsamp,)
            arg = arg + quad_term
        
        # Compute negative log-likelihood using PyTorch's numerically stable implementation
        # This replaces manual sigmoid + log computation with log-sum-exp trick for better stability
        nll = F.binary_cross_entropy_with_logits(arg, self.resp, reduction='mean')
        
        # Scale by fscale
        cost = self.fscale * nll
        
        # Add regularization terms
        reg_cost = self._compute_regularization(U, V, h)
        
        return cost + reg_cost

    def _compute_regularization(self, U, V, h=None):
        """Compute regularization penalty terms
        
        Args:
            U: first low-rank matrix or None
            V: second low-rank matrix or None
            h: linear weights vector or None
            
        Returns:
            torch tensor: scalar regularization cost
        """
        
        reg_cost = torch.tensor(0.0, dtype=self.float_dtype, device=self.device)
        
        # Nuclear-norm regularization for UV matrices
        if "nuclear-norm" in self.rtype and U is not None:
            reg_param = self.reg_params['nuclear-norm']
            
            # Nuclear norm regularization: λ * (||U||_F + ||V||_F)
            u_norm = torch.norm(U, p='fro')
            reg_cost = reg_cost + reg_param * u_norm
            
            if V is not None:
                v_norm = torch.norm(V, p='fro')
                reg_cost = reg_cost + reg_param * v_norm
        
        # L2-norm regularization for linear weights h (matching Theano implementation)
        if "l2-norm" in self.rtype and h is not None:
            l2_reg = self.reg_params['l2-norm']
            reg_cost += 0.5 * l2_reg * torch.sum(h ** 2)
        
        return reg_cost

    def initialize_weights(self, init_method='theano_compatible', **kwargs):
        """Initialize parameter vector
        
        Args:
            init_method: 'theano_compatible', 'zeros', 'small_random', 'random'
                - 'theano_compatible': matches original Theano initialization
                - 'zeros': all parameters start at zero
                - 'small_random': small random initialization (0.01 * randn)
                - 'random': standard random initialization
            seed: optional random seed for reproducibility
            
        Returns:
            numpy array: initialized parameter vector
        """
        print(f"Initializing weights with method: {init_method}")
        # Set random seed if provided
        seed = kwargs.get('seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if init_method == 'zeros':
            x = torch.zeros(self.param_size, dtype=self.float_dtype, device=self.device)
            
        elif init_method == 'small_random':
            x = 0.1 * torch.randn(self.param_size, dtype=self.float_dtype, device=self.device)
            
        elif init_method == 'theano_compatible':
            # Initialize like original Theano version
            x = torch.zeros(self.param_size, dtype=self.float_dtype, device=self.device)

            # Structured initialization for different parameter types
            idx = 0
            
            # Bias term: initialize to small negative value (like Theano)
            if self.use_vars['avar']:
                x[idx] = -0.1  # Small negative bias
                idx += 1
                
            # Linear weights: match true amplitude scale (0.1, not 0.001)
            if self.use_vars['hvar']:
                x[idx:idx+self.ndim] = 0.1 * torch.randn(self.ndim, dtype=self.float_dtype, device=self.device)
                idx += self.ndim
                
            # Low-rank parameters: orthogonal initialization
            if self.use_vars['UVvar']:
                if "UV-linear-insert" in self.cetype:
                    # Only U parameters needed
                    U_params = self.ndim * self.rank
                    U_init = torch.randn(self.ndim, self.rank, dtype=self.float_dtype, device=self.device)
                    # Orthogonalize and scale
                    U_init = torch.linalg.qr(U_init)[0] * 0.01
                    x[idx:idx+U_params] = U_init.reshape(-1)
                else:
                    # Both U and V parameters
                    UV_params = 2 * self.ndim * self.rank
                    x[idx:idx+UV_params] = 0.001 * torch.randn(UV_params, dtype=self.float_dtype, device=self.device)
                    
        else:  # 'random'
            x = torch.randn(self.param_size, dtype=self.float_dtype, device=self.device)
        
        return x.detach().cpu().numpy()

    def compute_gradient(self, x):
        """Compute gradient using PyTorch autograd
        
        Args:
            x: parameter vector tensor with requires_grad=True
            
        Returns:
            torch tensor: gradient vector
        """
        x.requires_grad_(True)
        cost = self.compute_cost(x)
        grad = torch.autograd.grad(cost, x)[0]
        return grad

    def predict_proba(self, feat_input, x=None):
        """Predict spike probabilities for given features
        
        Args:
            feat_input: input features tensor of shape (n_samples, n_features)
            x: parameter vector (if None, uses current parameters)
            
        Returns:
            numpy array: predicted probabilities of shape (n_samples,)
        """
        if x is None:
            # Use current parameters if available
            raise ValueError("Parameter vector x must be provided")
            
        # Convert parameters to weight matrices
        a, h, U, V = self.vec_to_weights(x)
        
        # Ensure feat_input is torch tensor
        if not isinstance(feat_input, torch.Tensor):
            feat_input = torch.tensor(feat_input, dtype=self.float_dtype, device=self.device)
        
        n_samples = feat_input.shape[0]
        
        # Initialize the argument of the logistic function
        if a is not None:
            arg = a.expand(n_samples)  # Broadcast to all samples
        else:
            arg = torch.zeros(n_samples, dtype=self.float_dtype, device=self.device)
        
        # Add linear term: h^T s
        if h is not None:
            linear_term = torch.matmul(feat_input, h)
            arg = arg + linear_term
        
        # Add quadratic term: s^T J s where J = U V^T
        if U is not None and V is not None:
            # Compute (U^T s) and (V^T s)
            Us = torch.matmul(feat_input, U)  # (n_samples, rank)
            Vs = torch.matmul(feat_input, V)  # (n_samples, rank)
            
            # Quadratic term: sum over rank of (U^T s) * (V^T s)
            quad_term = torch.sum(Us * Vs, dim=1)  # (n_samples,)
            arg = arg + quad_term
        
        # Compute probabilities
        probs = torch.sigmoid(arg)
        
        # Convert to numpy for compatibility
        return probs.detach().cpu().numpy()

    # Matrix conversion utilities (for compatibility)
    def UV_to_Q(self, U, V, **kwargs):
        """Convert U, V matrices to Q matrix"""
        return torch.cat([U, V], dim=0)
    
    def Q_to_UV(self, Q, **kwargs):
        """Convert Q matrix to U, V matrices"""
        U = Q[:self.ndim, :]
        V = Q[self.ndim:, :]
        return U, V
    
    def UV_to_J(self, U, V, **kwargs):
        """Convert U, V to interaction matrix J = U V^T"""
        return torch.matmul(U, V.T)
    
    def J_to_UV(self, J, **kwargs):
        """Decompose J matrix using SVD"""
        U_np, s_np, Vt_np = svd(J.detach().cpu().numpy())
        
        # Keep only top 'rank' components
        U = torch.tensor(U_np[:, :self.rank], dtype=self.float_dtype, device=self.device)
        s = torch.tensor(s_np[:self.rank], dtype=self.float_dtype, device=self.device)
        V = torch.tensor(Vt_np[:self.rank, :].T, dtype=self.float_dtype, device=self.device)
        
        # Scale by singular values
        U = U * torch.sqrt(s).unsqueeze(0)
        V = V * torch.sqrt(s).unsqueeze(0)
        
        return U, V
