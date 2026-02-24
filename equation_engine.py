"""
equation_engine.py
==================
SymPy-powered symbolic equation parser for Bayesian calibration.

Accepts any user-typed equation string like "y = a + b*x" and produces:
  - LaTeX rendering
  - PyMC model (for Bayesian MCMC)
  - NumPy-callable forward function
  - Symbolic or numerical inverse function

Also provides a VarianceModel class that parses a custom variance equation
of the form  ``sd = g(mu, <params>)``  where *mu* is the predicted mean and
any other symbol becomes a learnable parameter.
"""

import numpy as np
import sympy as sp
import re
from typing import Dict, List, Tuple, Optional


# ── Known function names that should NOT be treated as user symbols ───────
_KNOWN_FUNCTIONS = {
    "exp", "log", "sqrt", "sin", "cos", "tan",
    "sinh", "cosh", "tanh", "abs", "Abs",
}

# ── Known constants / reserved words ──────────────────────────────────────
_KNOWN_CONSTANTS = {"pi", "e", "E", "I", "N"}


def _pre_declare_symbols(rhs: str, local_dict: dict) -> dict:
    """Scan *rhs* for multi-character identifiers (e.g. ``b1``, ``alpha``)
    and add them to *local_dict* as SymPy Symbols **before** parsing.

    This prevents ``implicit_multiplication_application`` from splitting
    tokens like ``b1`` into ``b * 1`` or ``b2`` into ``b * 2``.

    Returns the updated *local_dict* (mutated in-place for convenience).
    """
    # Find every identifier: a letter (or _) followed by letters/digits/_
    tokens = set(re.findall(r'[A-Za-z_]\w*', rhs))
    # Remove anything already in local_dict, known functions, and constants
    already = set(local_dict.keys()) | _KNOWN_FUNCTIONS | _KNOWN_CONSTANTS
    for tok in tokens - already:
        local_dict[tok] = sp.Symbol(tok)
    return local_dict


# ══════════════════════════════════════════════════════════════════════════════
# Variance-model parser
# ══════════════════════════════════════════════════════════════════════════════

class VarianceModel:
    """Parse a user-supplied variance equation of the form
    ``sd = g(mu, ...)`` and produce a PyTensor-callable + LaTeX display.

    The variable ``mu`` refers to the predicted mean from the mean model.
    ``sigma`` is always implicitly available as the base noise scale.
    Every other symbol becomes a learnable variance parameter.
    """

    _RESERVED = {"x", "y", "pi", "e", "E", "I", "N"}

    def __init__(self, equation_str: str):
        self.equation_str = equation_str.strip()
        self._parse()
        self._build_numpy_callable()

    def _parse(self):
        s = self.equation_str
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs.lower() != "sd":
                raise ValueError(
                    f"Left-hand side must be 'sd', got '{lhs}'. "
                    f"Write: sd = g(mu, ...)"
                )
        else:
            rhs = s

        mu_sym = sp.Symbol("mu")
        sigma_sym = sp.Symbol("sigma")
        local_dict = {"mu": mu_sym, "sigma": sigma_sym,
                      "e": sp.E, "pi": sp.pi}
        local_dict = _pre_declare_symbols(rhs, local_dict)
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
                f"Could not parse variance equation: {rhs!r}\n"
                f"SymPy error: {exc}"
            ) from exc

        self.mu_sym = mu_sym
        self.sigma_sym = sigma_sym
        self.sd_sym = sp.Symbol("sd")

        all_symbols = sorted(self.rhs_expr.free_symbols, key=lambda s: s.name)
        # Variance-specific learnable parameters (everything except mu and sigma)
        self.param_symbols = [s for s in all_symbols
                              if s not in (mu_sym, sigma_sym)]
        self.param_names = [s.name for s in self.param_symbols]

        for pn in self.param_names:
            if pn in self._RESERVED:
                raise ValueError(
                    f"'{pn}' is reserved. Choose a different name. "
                    f"Reserved: {self._RESERVED}"
                )

        self.equation_sym = sp.Eq(self.sd_sym, self.rhs_expr)

    def _build_numpy_callable(self):
        """Build a NumPy-callable  sd = f(mu, sigma, *var_params)."""
        all_args = [self.mu_sym, self.sigma_sym] + self.param_symbols
        self._sd_lambda = sp.lambdify(all_args, self.rhs_expr, modules="numpy")

    def sd_numpy(self, mu: np.ndarray, sigma: np.ndarray,
                 var_params: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate sd = g(mu, sigma, ...) with NumPy arrays.

        Parameters
        ----------
        mu : array-like
            Predicted mean (or observed y used as a proxy).
        sigma : array-like
            Base noise scale draws.
        var_params : dict
            Draws for each variance-model parameter.

        Returns
        -------
        np.ndarray
            Standard-deviation values.
        """
        args = [mu, sigma] + [var_params[p] for p in self.param_names]
        return np.abs(self._sd_lambda(*args))

    # ── display ───────────────────────────────────────────────────────────
    def latex_str(self) -> str:
        return sp.latex(self.equation_sym)

    def variance_latex_str(self) -> str:
        """Return LaTeX for Var(y_i) = sd² form."""
        return (r"\mathrm{Var}(y_i) \;=\; \left("
                + sp.latex(self.rhs_expr) + r"\right)^2")

    def __repr__(self):
        return (f"VarianceModel('{self.equation_str}')  "
                f"params={self.param_names}")


# ══════════════════════════════════════════════════════════════════════════════
# Mean-model parser (original EquationModel)
# ══════════════════════════════════════════════════════════════════════════════

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
        local_dict = _pre_declare_symbols(rhs, local_dict)
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
        variance_eq: Optional[VarianceModel] = None,
    ):
        """Build and return a PyMC model for this equation.

        Parameters
        ----------
        x_cal, y_cal : array-like
            Calibration data.
        prior_config : dict, optional
            Per-parameter prior configuration.  Keys are parameter names
            (including 'sigma' and any variance-model parameters).
        log_scale_params : list[str], optional
            Parameter names modelled on the log scale.
        variance_model : str
            ``"constant"`` | ``"proportional"`` | ``"gelman2004"`` |
            ``"custom"``.
        variance_eq : VarianceModel, optional
            Required when *variance_model* is ``"custom"``.
        """
        import pymc as pm
        import pytensor.tensor as pt

        if prior_config is None:
            prior_config = {}
        if log_scale_params is None:
            log_scale_params = []

        _valid_vm = {"constant", "proportional", "gelman2004", "custom"}
        if variance_model not in _valid_vm:
            raise ValueError(
                f"variance_model must be one of {_valid_vm}, "
                f"got {variance_model!r}"
            )
        if variance_model == "custom" and variance_eq is None:
            raise ValueError(
                "variance_eq is required when variance_model='custom'"
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
                    log_cfg = cfg if cfg else {"dist": "Normal", "mu": 0, "sigma": 10}
                    log_var = _make_prior(f"log_{p_name}", log_cfg, pm)
                    param_vars[p_name] = pm.Deterministic(
                        p_name, pt.exp(log_var))
                else:
                    if not cfg:
                        cfg = {"dist": "Normal", "mu": 0, "sigma": 10}
                    param_vars[p_name] = _make_prior(p_name, cfg, pm)

            # Noise scale σ
            sigma_cfg = prior_config.get(
                "sigma", {"dist": "HalfNormal", "sigma": 10})
            sigma = _make_prior("sigma", sigma_cfg, pm)

            # Forward model mean
            args = [x_data] + [param_vars[p] for p in self.param_names]
            mu = pm.Deterministic("mu", pt_forward(*args))

            # ── Variance structure ────────────────────────────────────────
            if variance_model == "constant":
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_cal)

            elif variance_model == "proportional":
                sd = pt.abs(mu) * sigma
                sd = pt.maximum(sd, 1e-8)
                pm.Normal("y_obs", mu=mu, sigma=sd, observed=y_cal)

            elif variance_model == "gelman2004":
                A = float(np.exp(
                    np.mean(np.log(np.maximum(y_cal, 1e-12)))))
                alpha_cfg = prior_config.get(
                    "alpha",
                    {"dist": "Uniform", "lower": 0.0, "upper": 2.0},
                )
                alpha = _make_prior("alpha", alpha_cfg, pm)
                sd = pt.abs(mu / A) ** alpha * sigma
                sd = pt.maximum(sd, 1e-8)
                pm.Normal("y_obs", mu=mu, sigma=sd, observed=y_cal)

            elif variance_model == "custom":
                # Build pytensor callable for the variance equation
                var_param_vars = {}
                for vp in variance_eq.param_names:
                    vp_cfg = prior_config.get(vp, {})
                    if not vp_cfg:
                        vp_cfg = {"dist": "HalfNormal", "sigma": 2}
                    var_param_vars[vp] = _make_prior(vp, vp_cfg, pm)

                # lambdify:  sd = f(mu, sigma, <var_params...>)
                var_args = ([variance_eq.mu_sym, variance_eq.sigma_sym]
                            + variance_eq.param_symbols)
                pt_var_fn = sp.lambdify(
                    var_args, variance_eq.rhs_expr,
                    modules=[pt_mapping, "numpy"],
                )
                var_call_args = ([mu, sigma]
                                 + [var_param_vars[p]
                                    for p in variance_eq.param_names])
                sd = pt_var_fn(*var_call_args)
                sd = pt.maximum(pt.abs(sd), 1e-8)
                pm.Normal("y_obs", mu=mu, sigma=sd, observed=y_cal)

        return model

    def __repr__(self):
        return (f"EquationModel('{self.equation_str}')  "
                f"params={self.param_names}")


def _make_prior(name: str, cfg: Dict, pm_module):
    """Create a PyMC prior random variable from a config dict.

    Supported distributions: Normal, HalfNormal, Uniform, LogNormal,
    Exponential, Gamma.
    """
    dist = cfg.get("dist", "Normal")
    if dist == "Normal":
        return pm_module.Normal(
            name, mu=cfg.get("mu", 0), sigma=cfg.get("sigma", 10))
    elif dist == "HalfNormal":
        return pm_module.HalfNormal(name, sigma=cfg.get("sigma", 10))
    elif dist == "Uniform":
        return pm_module.Uniform(
            name, lower=cfg.get("lower", 0), upper=cfg.get("upper", 1))
    elif dist == "LogNormal":
        return pm_module.Lognormal(
            name, mu=cfg.get("mu", 0), sigma=cfg.get("sigma", 1))
    elif dist == "Exponential":
        return pm_module.Exponential(name, lam=cfg.get("lam", 1))
    elif dist == "Gamma":
        return pm_module.Gamma(
            name, alpha=cfg.get("alpha", 2), beta=cfg.get("beta", 1))
    else:
        raise ValueError(f"Unsupported prior distribution: {dist}")
