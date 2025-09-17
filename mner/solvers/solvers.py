from __future__ import print_function

import numpy as np
from scipy import optimize
import torch

""" solvers.py (module)

    This module contains the solver classes for mner-light-torch.
    PyTorch-native L-BFGS solver with automatic differentiation.

"""

class BaseSolver(object):
    """ BaseSolver (class)

        This class contains the basic necessities for optimizing a
        given objective function using iterative methods.

    """

    def __init__(self, **kwargs):
        """Initialize BaseSolver class instantiation.

            Initialize the class instantiation and some basic set-up
            of important parameters.

        """
        # get the floating point data type
        self.float_dtype_str = kwargs.get("float_dtype", "float64")
        self.float_dtype = getattr(torch, self.float_dtype_str)
        self.device = kwargs.get("device", "cpu")

    def solve(self, model, x0, **kwargs):
        """ solve (method)

            This method is the basic template for solving an
            optimization problem. This method should be overridden by
            child classes.

        """

        # cast the initial guess as a torch tensor with the right floating point data type
        x0 = torch.tensor(x0, dtype=self.float_dtype, device=self.device)

        return None


class LBFGSSolver(BaseSolver):
    """ LBFGSSolver (class)

        This class uses PyTorch's native L-BFGS optimizer to solve unconstrained optimization problems
        with automatic differentiation.

    """

    def __init__(self, **kwargs):
        """Initialize LBFGSSolver class instantiation.

            [inputs] (**kwargs)
                kwargs: keyword arguments are used to set
                  solver-specific hyperparameters.
                  - maxiter: (optional, default=500) maximum number
                    of iterations of the L-BFGS algorithm.
                  - maxfev: (optional, default=500) maximum number
                    of function evaluations of the L-BFGS algorithm.
                  - ftol: (optional, default=1e-6) function tolerance
                    of the L-BFGS algorithm.
                  - gtol: (optional, default=1e-6) gradient tolerance
                    of the L-BFGS algorithm.

        """

        # initialize the parent class
        super(LBFGSSolver, self).__init__(**kwargs)

        # get the maximum number of iterations
        self.maxiter = kwargs.get("maxiter", 500)
        print(f"  Using maxiter={self.maxiter}")
        # get the maximum number of function evaluations
        self.maxfev = kwargs.get("maxfev", 500)
        print(f"  Using maxfev={self.maxfev}")
        # get the function tolerance
        self.ftol = kwargs.get("ftol", 1e-6)
        print(f"  Using ftol={self.ftol}")
        # get the gradient tolerance
        self.gtol = kwargs.get("gtol", 1e-6)
        print(f"  Using gtol={self.gtol}")

    def solve(self, model, x0, **kwargs):
        """ solve (method)

            This method uses either PyTorch's native L-BFGS optimizer
            with PyTorch autograd to solve
            unconstrained optimization problems.

            [inputs] (model, x0, **kwargs)
                model: MNEr model instance with compute_cost method
                x0: numpy array or torch tensor that is the initial 
                  guess of the solution.
                kwargs: keyword arguments.

            [outputs] (xopt, fopt, results)
                xopt: numpy array that is the optimal solution.
                fopt: float that is the optimal objective value.
                results: dictionary that contains the results of the 
                  optimization.

        """

        # Convert initial guess to appropriate format
        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=self.float_dtype, device=self.device)
        elif isinstance(x0, torch.Tensor):
            x0 = x0.to(dtype=self.float_dtype, device=self.device)
        
        # Create parameter tensor
        params = x0.clone().detach().requires_grad_(True)
        
        # Create optimizer
        optimizer = torch.optim.LBFGS(
            [params], 
            max_iter=self.maxiter,
            tolerance_grad=self.gtol,
            tolerance_change=self.ftol,
            line_search_fn='strong_wolfe'
        )
        
        # Track optimization progress
        loss_history = []
        
        def closure():
            optimizer.zero_grad()
            # Project parameters onto constraint manifold
            with torch.no_grad():
                params.data = model.project_to_constraints(params.data)
            loss = model.compute_cost(params)
            loss.backward()
            loss_history.append(loss.item())
            return loss
        
        # Run optimization
        optimizer.step(closure)
        
        # Extract results
        xopt = params.detach().cpu().numpy()
        fopt = loss_history[-1] if loss_history else float('inf')
        
        results = {
            'success': True,
            'nit': len(loss_history),
            'nfev': len(loss_history),  # Approximate
            'loss_history': loss_history,
            'final_loss': fopt
        }
        
        return xopt, fopt, results
