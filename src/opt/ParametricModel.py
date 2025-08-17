import open3d as o3d
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize


class ParametricModel(ABC):

    def __init__(self):
        self.init_params = None
        self.optimized_params = None
    
    @abstractmethod
    def setup_optimization(self):
        """
        Give initial value to all the params, and keep in a turple in self.init_params.
        """
        pass

    @abstractmethod
    def _parametric_fn(self, x, y, params=None):# calculate samples over the parametric models.
        pass
    
    @abstractmethod
    def _objective_fn(self, params):# takes parameters as input to return result.
        pass

    @abstractmethod
    def _update_params(self, params=None):
        pass

    def _sample_xy(self, res):
        x = np.linspace(-0.5, 0.5, res)
        y = np.linspace(-0.5, 0.5, res)
        X, Y = np.meshgrid(x, y)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        return xy

    def optimize(self, method='L-BFGS-B'):
        options = {
            'ftol': 1e-8,
            'maxiter': 1000,   # Maximum number of iterations
            'disp': False,      # Display progress
        }

        result = minimize(self._objective_fn, self.init_params, method='Powell', options=options)
        # Update parameters after optimization
        if result.success:
            self._update_params(result.x)
            return result.x
        else:
            print("Optimization failed:", result.message)