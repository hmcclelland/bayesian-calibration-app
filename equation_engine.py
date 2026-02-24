"""

.py
==================
SymPy-powered symbolic equation parser for Bayesian calibration.

Accepts any user-typed equation string like "y = a + b*x" and produces:
  - LaTeX rendering
  - PyMC model (for Bayesian MCMC)
  - NumPy-callable forward function
  - Symbolic or numerical inverse function
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional


class EquationModel:
    """Parse a user-supplied equation string and produce everything needed
    for Bayesian calibration: PyMC model, NumPy forward/inverse functions."""

    _RESERVED = {"x", "y", "pi", "e", "E", "I", "N", "sigma"}

    def __init__(self, equation_str: str):
        self.equation_str = equation_str.strip()
        self._parse()

    def _parse(self):
        s = self.equation_str
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs.lower() != "y":
                raise ValueError(
                    f"Left-hand side must be 'y', got '{lhs}'. Write: y = f(x)"
                )
        else:
            rhs = s

        x = sp.Symbol("x")
        local_dict = {"x": x, "e": sp.E, "pi": sp.pi}
        transformations = (
            sp.parsing.sympy_parser.standard_transformations
            + (
                sp.parsing.sympy_parser.implicit_multiplication_application,
                sp.parsing.sympy_parser.convert_xor,
            )
        )
        try:
            self.rhs_expr = sp.parsing.sympy_parser.parse_expr(
                rhs, local_dict=local_dict, transformations=transformations,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not parse: {rhs!r}\nSymPy error: {exc}\n\n"
                f"Tip: use Python syntax, e.g.  a*exp(b*x)  or  a*x**b"
            ) from exc

        if x not in self.rhs_expr.free_symbols:
            raise ValueError(
                f"'x' does not appear in your equation: {rhs!r}\n"
                f"The model must depend on x."
            )

        self.x_sym = x
        self.y_sym = sp.Symbol("y")
        all_symbols = sorted(self.rhs_expr.free_symbols, key=lambda s: s.name)
        self.param_symbols = [s for s in all_symbols if s != x]
        self.param_names = [s.name for s in self.param_symbols]

        for pn in self.param_names:
            if pn in self._RESERVED:
                raise ValueError(
                    f"'{pn}' is reserved. Choose a different name. "
                    f"Reserved: {self._RESERVED}"
                )

        self.equation_sym = sp.Eq(self.y_sym, self.rhs_expr)
        self._build_inverse()
        self._build_numpy_functions()

    # ---- display ---------------------------------------------------------
    def latex_str(self) -> str:
        return sp.latex(self.equation_sym)

    def inverse_latex_str(self) -> Optional[str]:
        if self._has_symbolic_inverse:
            return sp.latex(sp.Eq(self.x_sym, self._inverse_exprs[0]))
        return None

    # ---- inverse ---------------------------------------------------------
    def _build_inverse(self):
        try:
            solutions = sp.solve(self.equation_sym, self.x_sym)
            real_solutions = [
                sol for sol in solutions if sol.is_real is not False
            ]
            if real_solutions:
                self._inverse_exprs = real_solutions
                self._has_symbolic_inverse = True
            else:
                self._inverse_exprs = []
                self._has_symbolic_inverse = False
        except Exception:
            self._inverse_exprs = []
            self._has_symbolic_inverse = False

    @property
    def has_symbolic_inverse(self) -> bool:
        return self._has_symbolic_inverse

    # ---- numpy callables -------------------------------------------------
    def _build_numpy_functions(self):
        all_args = [self.x_sym] + self.param_symbols
        self._forward_lambda = sp.lambdify(
            all_args, self.rhs_expr, modules="numpy"
        )
        self._inverse_lambdas = []
        if self._has_symbolic_inverse:
            inv_args = [self.y_sym] + self.param_symbols
            for sol in self._inverse_exprs:
                self._inverse_lambdas.append(
                    sp.lambdify(inv_args, sol, modules="numpy")
                )

    def forward_numpy(
        self, params: Dict[str, np.ndarray], x: np.ndarray
    ) -> np.ndarray:
        args = [x] + [params[p] for p in self.param_names]
        return self._forward_lambda(*args)

    def inverse_numpy(
        self,
        y_val,
        params: Dict[str, np.ndarray],
        x_hint: float = 1.0,
        x_range: Tuple[float, float] = (-1e6, 1e6),
    ) -> np.ndarray:
        if self._has_symbolic_inverse:
            return self._inverse_symbolic(y_val, params, x_hint)
        else:
            return self._inverse_numerical(y_val, params, x_range)

    def _inverse_symbolic(
        self, y_val, params: Dict[str, np.ndarray], x_hint: float
    ) -> np.ndarray:
        args = [y_val] + [params[p] for p in self.param_names]
        if len(self._inverse_lambdas) == 1:
            result = self._inverse_lambdas[0](*args)
            if np.iscomplexobj(result):
                result = np.real(result)
            return result
        else:
            candidates = []
            for fn in self._inverse_lambdas:
                r = fn(*args)
                if np.iscomplexobj(r):
                    r = np.real(r)
                candidates.append(r)
            candidates = np.array(candidates)
            dists = np.abs(candidates - x_hint)
            best_idx = np.argmin(dists, axis=0)
            n_draws = candidates.shape[1] if candidates.ndim > 1 else 1
            return candidates[best_idx, np.arange(n_draws)]

    def _inverse_numerical(
        self,
        y_val,
        params: Dict[str, np.ndarray],
        x_range: Tuple[float, float],
    ) -> np.ndarray:
        from scipy.optimize import brentq

        y_val = np.atleast_1d(np.asarray(y_val, dtype=float))
        n = len(y_val)
        result = np.full(n, np.nan)

        for i in range(n):
            p_i = {
                k: v[i] if np.ndim(v) > 0 else v for k, v in params.items()
            }
            target = float(y_val[i])

            def residual(x_try, _target=target, _p=p_i):
                return (
                    float(self.forward_numpy(_p, np.array([x_try]))[0])
                    - _target
                )

            try:
                result[i] = brentq(
                    residual, x_range[0], x_range[1], xtol=1e-10
                )
            except (ValueError, RuntimeError):
                result[i] = np.nan

        return result

    # ---- PyMC model builder ----------------------------------------------
    def build_pymc_model(self, x_cal, y_cal):
        """Build and return a PyMC model for this equation.

        Parameters get Normal(0, 10) priors; sigma gets HalfNormal(10).
        """
        import pymc as pm
        import pytensor.tensor as pt

        # Build a pytensor-compatible forward function from SymPy
        pt_mapping = {
            "exp": pt.exp, "log": pt.log, "sqrt": pt.sqrt,
            "sin": pt.sin, "cos": pt.cos, "tan": pt.tan,
            "sinh": pt.sinh, "cosh": pt.cosh, "tanh": pt.tanh,
            "Abs": pt.abs,
        }
        all_args = [self.x_sym] + self.param_symbols
        pt_forward = sp.lambdify(
            all_args, self.rhs_expr,
            modules=[pt_mapping, "numpy"],
        )

        with pm.Model() as model:
            x_data = pm.Data("x_data", x_cal)

            param_vars = {}
            for p_name in self.param_names:
                param_vars[p_name] = pm.Normal(p_name, mu=0, sigma=10)

            sigma = pm.HalfNormal("sigma", sigma=10)

            args = [x_data] + [param_vars[p] for p in self.param_names]
            mu = pm.Deterministic("mu", pt_forward(*args))

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_cal)

        return model

    def build_pymc_model_heteroscedastic(self, x_cal, y_cal, variance_model="linear"):
        """Build a PyMC model with observation-level variance.

        variance_model:
            "linear"  — sigma_i = sigma0 + sigma1 * |mu_i|
            "power"   — sigma_i = sigma0 * |mu_i|^delta
        """
        import pymc as pm
        import pytensor.tensor as pt

        pt_mapping = {
            "exp": pt.exp, "log": pt.log, "sqrt": pt.sqrt,
            "sin": pt.sin, "cos": pt.cos, "tan": pt.tan,
            "sinh": pt.sinh, "cosh": pt.cosh, "tanh": pt.tanh,
            "Abs": pt.abs,
        }
        all_args = [self.x_sym] + self.param_symbols
        pt_forward = sp.lambdify(
            all_args, self.rhs_expr,
            modules=[pt_mapping, "numpy"],
        )

        with pm.Model() as model:
            x_data = pm.Data("x_data", x_cal)

            param_vars = {}
            for p_name in self.param_names:
                param_vars[p_name] = pm.Normal(p_name, mu=0, sigma=10)

            args = [x_data] + [param_vars[p] for p in self.param_names]
            mu = pm.Deterministic("mu", pt_forward(*args))

            abs_mu = pt.abs(mu) + 1e-8          # guard against zero

            if variance_model == "linear":
                sigma0 = pm.HalfNormal("sigma0", sigma=10)
                sigma1 = pm.HalfNormal("sigma1", sigma=1)
                sigma = pm.Deterministic("sigma", sigma0 + sigma1 * abs_mu)
            elif variance_model == "power":
                sigma0 = pm.HalfNormal("sigma0", sigma=10)
                delta  = pm.HalfNormal("delta", sigma=1)
                sigma  = pm.Deterministic("sigma", sigma0 * pt.pow(abs_mu, delta))
            else:
                raise ValueError(f"Unknown variance_model: {variance_model!r}")

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_cal)

        return model

    def __repr__(self):
        return f"EquationModel('{self.equation_str}')  params={self.param_names}"
