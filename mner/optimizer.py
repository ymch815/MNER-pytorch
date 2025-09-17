import numpy as np
import torch
from . import model
from .solvers.solvers import LBFGSSolver
from .util import util

""" optimizer.py (module)

    The optimizer module contains the PyTorch-based Optimizer class 
    that is used to configure low-rank MNE models, initialize the solver, 
    and minimize the objective function using automatic differentiation.

"""

class Optimizer(object):
    """ Optimizer (class)

        The Optimizer class is the interface used for constructing
        and optimizing low-rank MNE models with PyTorch backend.

    """

    def __init__(self, resp, feat, rank, cetype=None, citype=None, rtype=None, solver=None, datasets=None, **kwargs):
        """Initialize Optimizer class instantiation.

            [inputs] (resp, feat, rank, cetype=None, citype=None,
              rtype=None, solver=None, datasets=None, **kwargs)
                resp: numpy array of the output labels with shape
                  (nsamp,) where nsamp is the number of data samples.
                feat: numpy array of the input features with shape
                  (nsamp, ndim) where ndim is the number of features.
                rank: positive integer that sets the number of columns
                  of the matrices U and V.
                cetype: (optional) list of equality constraints.
                  Supported: ["UV-linear-insert"]
                citype: (optional) inequality constraints (not supported)
                rtype: (optional) list of regularization penalties.
                  Supported: ["nuclear-norm"]
                solver: (optional) solver instance. Defaults to LBFGSSolver
                datasets: (optional) dictionary with train/cv/test indices
                kwargs: keyword arguments including device, dtype settings

        """

        # Initialize class members to standard arguments
        self.rank = rank
        self.cetype = cetype
        self.citype = citype
        self.rtype = rtype
        self.datasets = datasets

        # Device and dtype settings
        self.device = kwargs.get('device', 'cpu')
        self.float_dtype = kwargs.get('float_dtype', 'float64')

        # Get data sizes
        self.nsamp, self.ndim = self.get_data_sizes(feat)
        self.ntrain, self.ncv, self.ntest = self.get_data_subset_sample_sizes(self.nsamp, self.datasets)

        # Store data
        self.resp_full = resp
        self.feat_full = feat

        # Initialize solver
        if solver is None:
            # Remove device and float_dtype from kwargs to avoid conflicts
            solver_kwargs = {k: v for k, v in kwargs.items() if k not in ['device', 'float_dtype']}
            self.solver = LBFGSSolver(device=self.device, float_dtype=self.float_dtype, **solver_kwargs)
        else:
            self.solver = solver

        # Store additional arguments
        self.kwargs = kwargs

    def get_data_sizes(self, feat):
        """Get the number of samples and features from the feature array"""
        nsamp, ndim = feat.shape
        return nsamp, ndim

    def get_data_subset_sample_sizes(self, nsamp, datasets):
        """Get the number of training, cross-validation, and test samples"""
        
        if datasets is None:
            # All samples are training samples
            return nsamp, 0, 0
        
        ntrain = np.sum(datasets.get("trainset", np.ones(nsamp, dtype=bool)))
        ncv = np.sum(datasets.get("cvset", np.zeros(nsamp, dtype=bool)))
        ntest = np.sum(datasets.get("testset", np.zeros(nsamp, dtype=bool)))
        
        return ntrain, ncv, ntest

    def config_models(self):
        """Configure model instances for training, cross-validation, and testing"""
        
        # Remove device and float_dtype from kwargs to avoid conflicts
        model_kwargs = {k: v for k, v in self.kwargs.items() if k not in ['device', 'float_dtype']}
        
        if self.datasets is None:
            # Single model for all data
            self.train_model = model.MNEr(
                self.resp_full, self.feat_full, self.rank,
                cetype=self.cetype, citype=self.citype, rtype=self.rtype,
                device=self.device, float_dtype=self.float_dtype,
                **model_kwargs
            )
            self.cv_model = None
            self.test_model = None
        else:
            # Separate models for each dataset split
            trainset = self.datasets.get("trainset", np.ones(self.nsamp, dtype=bool))
            cvset = self.datasets.get("cvset", np.zeros(self.nsamp, dtype=bool))
            testset = self.datasets.get("testset", np.zeros(self.nsamp, dtype=bool))

            # Training model
            if np.any(trainset):
                self.train_model = model.MNEr(
                    self.resp_full[trainset], self.feat_full[trainset, :], self.rank,
                    cetype=self.cetype, citype=self.citype, rtype=self.rtype,
                    device=self.device, float_dtype=self.float_dtype,
                    **model_kwargs
                )
            else:
                self.train_model = None

            # Cross-validation model
            if np.any(cvset):
                self.cv_model = model.MNEr(
                    self.resp_full[cvset], self.feat_full[cvset, :], self.rank,
                    cetype=self.cetype, citype=self.citype, rtype=self.rtype,
                    device=self.device, float_dtype=self.float_dtype,
                    **model_kwargs
                )
            else:
                self.cv_model = None

            # Test model
            if np.any(testset):
                self.test_model = model.MNEr(
                    self.resp_full[testset], self.feat_full[testset, :], self.rank,
                    cetype=self.cetype, citype=self.citype, rtype=self.rtype,
                    device=self.device, float_dtype=self.float_dtype,
                    **model_kwargs
                )
            else:
                self.test_model = None

    def optimize(self, x0=None, **kwargs):
        """Run optimization

            [inputs] (x0=None, **kwargs)
                x0: initial parameter guess. If None, random initialization
                kwargs: additional solver arguments

            [outputs] (results)
                results: dictionary containing optimization results
        """

        # Configure models if not already done
        if not hasattr(self, 'train_model'):
            self.config_models()

        # Use training model for optimization
        if self.train_model is None:
            raise ValueError("No training data available")

        # Initialize parameters if not provided
        if x0 is None:
            x0 = self.train_model.initialize_weights(**kwargs)

        # Run optimization
        xopt, fopt, opt_info = self.solver.solve(self.train_model, x0, **kwargs)

        # Prepare results dictionary
        results = {
            'xopt': xopt,
            'fopt': fopt,
            'optimization_info': opt_info,
            'model': self.train_model,
            'param_size': self.train_model.get_param_size()
        }

        # Evaluate on other datasets if available
        if self.cv_model is not None:
            x_torch = torch.tensor(xopt, dtype=getattr(torch, self.float_dtype), device=self.device)
            cv_cost = self.cv_model.compute_cost(x_torch).item()
            results['cv_cost'] = cv_cost

        if self.test_model is not None:
            x_torch = torch.tensor(xopt, dtype=getattr(torch, self.float_dtype), device=self.device)
            test_cost = self.test_model.compute_cost(x_torch).item()
            results['test_cost'] = test_cost

        return results

    def evaluate_model(self, x, model_type='train'):
        """Evaluate model at given parameters
        
        Args:
            x: parameter vector (numpy array or torch tensor)
            model_type: 'train', 'cv', or 'test'
            
        Returns:
            dict: evaluation results including cost, predictions, etc.
        """
        
        # Select model
        if model_type == 'train':
            eval_model = getattr(self, 'train_model', None)
        elif model_type == 'cv':
            eval_model = getattr(self, 'cv_model', None)
        elif model_type == 'test':
            eval_model = getattr(self, 'test_model', None)
        else:
            raise ValueError("model_type must be 'train', 'cv', or 'test'")

        if eval_model is None:
            raise ValueError(f"No {model_type} model available")

        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x_torch = torch.tensor(x, dtype=getattr(torch, self.float_dtype), device=self.device)
        else:
            x_torch = x

        # Compute cost
        cost = eval_model.compute_cost(x_torch).item()

        # Get weight matrices
        a, h, U, V = eval_model.vec_to_weights(x_torch)

        # Compute predictions (probabilities)
        with torch.no_grad():
            if a is not None:
                arg = a.expand(eval_model.nsamp)
            else:
                arg = torch.zeros(eval_model.nsamp, dtype=eval_model.float_dtype, device=eval_model.device)
            
            if h is not None:
                arg = arg + torch.matmul(eval_model.feat, h)
            
            if U is not None and V is not None:
                Us = torch.matmul(eval_model.feat, U)
                Vs = torch.matmul(eval_model.feat, V)
                quad_term = torch.sum(Us * Vs, dim=1)
                arg = arg + quad_term
            
            predictions = torch.sigmoid(arg).cpu().numpy()

        results = {
            'cost': cost,
            'predictions': predictions,
            'weights': {
                'a': a.cpu().numpy() if a is not None else None,
                'h': h.cpu().numpy() if h is not None else None,
                'U': U.cpu().numpy() if U is not None else None,
                'V': V.cpu().numpy() if V is not None else None
            }
        }

        return results
