"""
equation_engine.py
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
    def build_pymc_model(
        self,
        x_cal,
        y_cal,
        prior_config: Optional[Dict] = None,
        log_scale_params: Optional[List[str]] = None,
        variance_model: str = "constant",
    ):
        """Build and return a PyMC model for this equation.

        Parameters
        ----------
        x_cal, y_cal : array-like
            Calibration data.
        prior_config : dict, optional
            Per-parameter prior configuration.  Keys are parameter names
            (including 'sigma' and optionally 'alpha').  Each value is a dict
            with keys 'dist' ('Normal', 'HalfNormal', 'Uniform', 'LogNormal'),
            and distribution-specific keyword arguments (e.g. mu, sigma, lower,
            upper).  If a parameter is absent, sensible defaults are used.
        log_scale_params : list[str], optional
            Parameter names that should be modelled on the log scale
            (i.e. the prior is placed on log(param) and the actual value
            used in the forward model is exp(log_param)).  This enforces
            positivity.
        variance_model : str
            One of:
            - ``"constant"`` — homoscedastic, Var(y_i) = σ².
            - ``"proportional"`` — constant-CV, Var(y_i) = μ_i² · σ².
            - ``"gelman2004"`` — Gelman, Chew & Shnaidman (2004):
              Var(y_i) = (μ_i / A)^{2α} · σ²  with learnable α.
        """
        import pymc as pm
        import pytensor.tensor as pt

        if prior_config is None:
            prior_config = {}
        if log_scale_params is None:
            log_scale_params = []

        _valid_vm = {"constant", "proportional", "gelman2004"}
        if variance_model not in _valid_vm:
            raise ValueError(
                f"variance_model must be one of {_valid_vm}, got {variance_model!r}"
            )

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
                cfg = prior_config.get(p_name, {})
                if p_name in log_scale_params:
                    # Prior is on log(param); actual param = exp(log_param)
                    log_cfg = cfg if cfg else {"dist": "Normal", "mu": 0, "sigma": 10}
                    log_var = _make_prior(f"log_{p_name}", log_cfg, pm)
                    param_vars[p_name] = pm.Deterministic(p_name, pt.exp(log_var))
                else:
                    if not cfg:
                        cfg = {"dist": "Normal", "mu": 0, "sigma": 10}
                    param_vars[p_name] = _make_prior(p_name, cfg, pm)

            # Noise scale σ_y
            sigma_cfg = prior_config.get("sigma", {"dist": "HalfNormal", "sigma": 10})
            sigma = _make_prior("sigma", sigma_cfg, pm)

            # Forward model mean
            args = [x_data] + [param_vars[p] for p in self.param_names]
            mu = pm.Deterministic("mu", pt_forward(*args))

            if variance_model == "constant":
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_cal)

            elif variance_model == "proportional":
                # Constant-CV model: sd_i = |μ_i| · σ_y
                sd = pt.abs(mu) * sigma
                sd = pt.maximum(sd, 1e-8)
                pm.Normal("y_obs", mu=mu, sigma=sd, observed=y_cal)

            elif variance_model == "gelman2004":
                # Gelman et al. (2004) variance model:
                #   Var(y_i) = (μ_i / A)^(2α) · σ_y²
                # A = geometric mean of observed y (constant)
                A = float(np.exp(np.mean(np.log(np.maximum(y_cal, 1e-12)))))
                alpha_cfg = prior_config.get(
                    "alpha", {"dist": "Uniform", "lower": 0.0, "upper": 2.0}
                )
                alpha = _make_prior("alpha", alpha_cfg, pm)
                # sd_i = |μ_i / A|^α · σ_y   (use abs to handle edge cases)
                sd = pt.abs(mu / A) ** alpha * sigma
                sd = pt.maximum(sd, 1e-8)
                pm.Normal("y_obs", mu=mu, sigma=sd, observed=y_cal)

        return model

    def __repr__(self):
        return f"EquationModel('{self.equation_str}')  params={self.param_names}"


def _make_prior(name: str, cfg: Dict, pm_module):
    """Create a PyMC prior random variable from a config dict.

    Supported distributions: Normal, HalfNormal, Uniform, LogNormal.
    """
    dist = cfg.get("dist", "Normal")
    if dist == "Normal":
        return pm_module.Normal(name, mu=cfg.get("mu", 0), sigma=cfg.get("sigma", 10))
    elif dist == "HalfNormal":
        return pm_module.HalfNormal(name, sigma=cfg.get("sigma", 10))
    elif dist == "Uniform":
        return pm_module.Uniform(name, lower=cfg.get("lower", 0), upper=cfg.get("upper", 1))
    elif dist == "LogNormal":
        return pm_module.Lognormal(name, mu=cfg.get("mu", 0), sigma=cfg.get("sigma", 1))
    else:
        raise ValueError(f"Unsupported prior distribution: {dist}")
