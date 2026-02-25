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
    Every other symbol becomes a learnable variance parameter — **unless**
    it is wrapped in square brackets (e.g. ``[A]``), in which case it is
    treated as a *prescribed* (user-supplied constant) parameter.
    """

    _RESERVED = {"y", "pi", "e", "E", "I", "N"}

    def __init__(self, equation_str: str):
        self.equation_str = equation_str.strip()
        self._parse()
        self._build_numpy_callable()

    def _parse(self):
        s = self.equation_str

        # ── Detect prescribed parameters (square brackets) ────────────
        # Find tokens like  [A]  or  [myConst]  — an identifier wrapped
        # in square brackets.
        self._prescribed_raw: List[str] = re.findall(
            r'\[([A-Za-z_]\w*)\]', s)
        # Strip the brackets from the equation string so SymPy can parse it
        _clean = re.sub(r'\[([A-Za-z_]\w*)\]', r'\1', s)

        if "=" in _clean:
            lhs, rhs = _clean.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs.lower() != "sd":
                raise ValueError(
                    f"Left-hand side must be 'sd', got '{lhs}'. "
                    f"Write: sd = g(mu, ...)"
                )
        else:
            rhs = _clean

        mu_sym = sp.Symbol("mu")
        sigma_sym = sp.Symbol("sigma")
        x_sym = sp.Symbol("x")
        local_dict = {"mu": mu_sym, "sigma": sigma_sym, "x": x_sym,
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
        self.x_sym = x_sym
        self.sd_sym = sp.Symbol("sd")

        all_symbols = sorted(self.rhs_expr.free_symbols, key=lambda s: s.name)
        self.uses_x = x_sym in self.rhs_expr.free_symbols

        # Separate prescribed symbols from learnable symbols
        prescribed_set = set(self._prescribed_raw)
        self.prescribed_symbols = [s for s in all_symbols
                                   if s.name in prescribed_set
                                   and s not in (mu_sym, sigma_sym, x_sym)]
        self.prescribed_names = [s.name for s in self.prescribed_symbols]

        # Variance-specific learnable parameters (everything except mu,
        # sigma, x, and prescribed params)
        self.param_symbols = [s for s in all_symbols
                              if s not in (mu_sym, sigma_sym, x_sym)
                              and s.name not in prescribed_set]
        self.param_names = [s.name for s in self.param_symbols]

        for pn in self.param_names + self.prescribed_names:
            if pn in self._RESERVED:
                raise ValueError(
                    f"'{pn}' is reserved. Choose a different name. "
                    f"Reserved: {self._RESERVED}"
                )

        self.equation_sym = sp.Eq(self.sd_sym, self.rhs_expr)

    def _build_numpy_callable(self):
        """Build a NumPy-callable  sd = f(x, mu, sigma, *prescribed, *var_params)."""
        all_args = (([self.x_sym] if self.uses_x else [])
                    + [self.mu_sym, self.sigma_sym]
                    + self.prescribed_symbols + self.param_symbols)
        self._sd_lambda = sp.lambdify(all_args, self.rhs_expr, modules="numpy")

    def sd_numpy(self, mu: np.ndarray, sigma: np.ndarray,
                 var_params: Dict[str, np.ndarray],
                 prescribed_params: Optional[Dict[str, float]] = None,
                 x: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        """Evaluate sd = g(mu, sigma, ...) with NumPy arrays.

        Parameters
        ----------
        mu : array-like
            Predicted mean (or observed y used as a proxy).
        sigma : array-like
            Base noise scale draws.
        var_params : dict
            Draws for each *learnable* variance-model parameter.
        prescribed_params : dict, optional
            Fixed values for prescribed (``[A]``-wrapped) parameters.

        Returns
        -------
        np.ndarray
            Standard-deviation values (always real, non-negative).
        """
        if prescribed_params is None:
            prescribed_params = {}

        # Ensure all inputs are NumPy arrays for consistent broadcasting
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        prescribed_args = [float(prescribed_params.get(p, 1.0))
                           for p in self.prescribed_names]
        vp_arrays = [np.asarray(var_params[p], dtype=float)
                     for p in self.param_names]
        x_args = [np.asarray(x, dtype=float)] if self.uses_x else []
        args = x_args + [mu, sigma] + prescribed_args + vp_arrays

        try:
            result = self._sd_lambda(*args)
        except (TypeError, ValueError, ZeroDivisionError):
            # Fallback: if the lambdified function fails (e.g. negative
            # base with fractional exponent), return |sigma| as a safe
            # constant-noise fallback.
            return np.abs(sigma)

        # The lambdified expression may return complex values when mu < 0
        # and the equation involves fractional powers.  Take the real
        # magnitude and ensure the output is a finite float array.
        result = np.asarray(result)
        if np.iscomplexobj(result):
            result = np.abs(result).real
        else:
            result = np.abs(result)

        # Replace any NaN / Inf with a safe fallback (|sigma|)
        bad = ~np.isfinite(result)
        if np.any(bad):
            result[bad] = np.abs(sigma) if np.ndim(sigma) == 0 else np.abs(sigma)[bad] if np.shape(sigma) == np.shape(result) else np.mean(np.abs(sigma))

        return result

    # ── display ───────────────────────────────────────────────────────────
    def latex_str(self) -> str:
        return sp.latex(self.equation_sym)

    def variance_latex_str(self) -> str:
        """Return LaTeX for Var(y_i) = sd² form."""
        return (r"\mathrm{Var}(y_i) \;=\; \left("
                + sp.latex(self.rhs_expr) + r"\right)^2")

    def __repr__(self):
        return (f"VarianceModel('{self.equation_str}')  "
                f"params={self.param_names}  "
                f"prescribed={self.prescribed_names}")


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
        prescribed_values: Optional[Dict[str, float]] = None,
    ):
        """Build and return a PyMC model for this equation.

        Parameters
        ----------
        x_cal, y_cal : array-like
            Calibration data.
        prior_config : dict, optional
            Per-parameter prior configuration.  Keys are parameter names
            (including 'sigma' and any variance-model parameters).
            For log-scale parameters, the config should describe the prior
            in the **original** (natural) parameter space — it will be
            automatically converted to log-space internally.
        log_scale_params : list[str], optional
            Parameter names modelled on the log scale.
        variance_model : str
            ``"constant"`` | ``"proportional"`` | ``"gelman2004"`` |
            ``"custom"``.
        variance_eq : VarianceModel, optional
            Required when *variance_model* is ``"custom"``.
        prescribed_values : dict, optional
            Fixed values for prescribed (``[A]``-wrapped) parameters in the
            variance equation.  Keys are bare names (without ``[ ]``).

        Returns
        -------
        tuple[pm.Model, dict | None]
            The PyMC model and an optional dict of initial values (keyed by
            user-facing parameter names) that should be passed to
            ``pm.sample(initvals=...)``.
        """
        import pymc as pm
        import pytensor.tensor as pt

        if prior_config is None:
            prior_config = {}
        if log_scale_params is None:
            log_scale_params = []
        if prescribed_values is None:
            prescribed_values = {}

        # ── Compute least-squares starting values for initialisation ──────
        ls_initvals = self._estimate_initvals(x_cal, y_cal)

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

        # ── Prepare initvals dict (user-facing names only) ────────────────
        sample_initvals: Optional[Dict[str, float]] = None

        with pm.Model() as model:
            x_data = pm.Data("x_data", x_cal)

            param_vars = {}
            for p_name in self.param_names:
                cfg = prior_config.get(p_name, {})
                if p_name in log_scale_params:
                    # Convert the prior config to log-space.
                    # The user specifies priors in the ORIGINAL parameter
                    # space; we need a prior on log(param).
                    # Only Normal priors on the log-space are safe — any
                    # positivity-constrained dist (HalfNormal, LogNormal,
                    # Gamma, Exponential) would cause PyMC to add ANOTHER
                    # internal log-transform on top of our manual exp(),
                    # resulting in a double-transform → inf.
                    log_cfg = self._convert_prior_to_log_space(cfg, p_name, ls_initvals)
                    log_var = _make_prior(f"log_{p_name}", log_cfg, pm)
                    param_vars[p_name] = pm.Deterministic(
                        p_name, pt.exp(log_var))

                    # Set initval for log-scale param
                    if ls_initvals and p_name in ls_initvals:
                        val = ls_initvals[p_name]
                        if val > 0:
                            if sample_initvals is None:
                                sample_initvals = {}
                            sample_initvals[f"log_{p_name}"] = float(np.log(val))
                else:
                    if not cfg:
                        cfg = {"dist": "Normal", "mu": 0, "sigma": 100}
                    param_vars[p_name] = _make_prior(p_name, cfg, pm)

                    # Set initval for regular param
                    if ls_initvals and p_name in ls_initvals:
                        if sample_initvals is None:
                            sample_initvals = {}
                        sample_initvals[p_name] = float(ls_initvals[p_name])

            # Noise scale σ
            sigma_cfg = prior_config.get(
                "sigma", {"dist": "Uniform", "lower": 0.0, "upper": 50.0})
            sigma = _make_prior("sigma", sigma_cfg, pm)

            # Set initval for sigma from LS residuals
            if ls_initvals:
                try:
                    ls_pred = self.forward_numpy(
                        ls_initvals, np.asarray(x_cal, dtype=float))
                    ls_resid_sd = float(
                        np.std(np.asarray(y_cal, dtype=float) - ls_pred))
                    if ls_resid_sd > 0:
                        if sample_initvals is None:
                            sample_initvals = {}
                        sample_initvals["sigma"] = ls_resid_sd
                except Exception:
                    pass

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

                # lambdify includes x (if used) and prescribed symbols as arguments
                var_args = (([variance_eq.x_sym] if variance_eq.uses_x else [])
                            + [variance_eq.mu_sym, variance_eq.sigma_sym]
                            + variance_eq.prescribed_symbols
                            + variance_eq.param_symbols)
                pt_var_fn = sp.lambdify(
                    var_args, variance_eq.rhs_expr,
                    modules=[pt_mapping, "numpy"],
                )
                # Prescribed params become fixed float constants
                prescribed_call_args = [
                    float(prescribed_values.get(p, 1.0))
                    for p in variance_eq.prescribed_names
                ]
                var_call_args = (([x_data] if variance_eq.uses_x else [])
                                 + [mu, sigma]
                                 + prescribed_call_args
                                 + [var_param_vars[p]
                                    for p in variance_eq.param_names])
                sd = pt_var_fn(*var_call_args)
                sd = pt.maximum(pt.abs(sd), 1e-8)
                pm.Normal("y_obs", mu=mu, sigma=sd, observed=y_cal)

        return model, sample_initvals

    @staticmethod
    def _convert_prior_to_log_space(
        cfg: Dict, p_name: str, ls_initvals: Optional[Dict[str, float]]
    ) -> Dict:
        """Convert a prior config from original parameter space to log-space.

        When the user enables log-scale for a parameter, the model uses:
            log_p ~ Prior(...)
            p = exp(log_p)

        The prior on log_p MUST be an unbounded distribution (Normal or
        Uniform with wide bounds). Using a positivity-constrained dist
        (HalfNormal, LogNormal, Gamma, Exponential) on log_p would cause
        PyMC to add an internal log-transform on top of our manual exp(),
        leading to a double-transform and inf starting values.

        This method:
        1. Forces the distribution to Normal (safe, unbounded).
        2. Converts mu/sigma from original space to log-space using the
           delta method: if p ~ Normal(μ, σ), then log(p) ≈ Normal(log(μ), σ/μ).
        3. Falls back to LS estimates or generic defaults if needed.
        """
        dist = cfg.get("dist", "Normal") if cfg else "Normal"

        # If user explicitly chose Normal, convert its parameters to log-space
        if dist == "Normal" and cfg:
            mu_orig = cfg.get("mu", 0.0)
            sigma_orig = cfg.get("sigma", 10.0)
            if mu_orig > 0:
                # Delta method: log(p) ≈ Normal(log(μ), σ/μ)
                log_mu = float(np.log(mu_orig))
                log_sigma = max(sigma_orig / mu_orig, 0.5)
                return {"dist": "Normal",
                        "mu": round(log_mu, 4),
                        "sigma": round(log_sigma, 2)}

        # For any positivity-constrained dist (HalfNormal, LogNormal,
        # Gamma, Exponential), we CANNOT use it on log-space safely.
        # Convert to a Normal in log-space using LS estimate as centre.
        if ls_initvals and p_name in ls_initvals:
            val = ls_initvals[p_name]
            if val > 0:
                return {"dist": "Normal",
                        "mu": round(float(np.log(val)), 4),
                        "sigma": 2.0}

        # Ultimate fallback: vague Normal on log-space
        return {"dist": "Normal", "mu": 0.0, "sigma": 3.0}

    def _estimate_initvals(
        self, x_cal, y_cal
    ) -> Optional[Dict[str, float]]:
        """Try scipy.optimize.curve_fit to get least-squares starting values.

        Returns a dict {param_name: value} or None if it fails.
        """
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            return None

        x_cal = np.asarray(x_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)

        def _wrapper(x, *args):
            p = {name: args[i] for i, name in enumerate(self.param_names)}
            return self.forward_numpy(p, x)

        # Generate multiple initial guesses to increase chance of convergence
        n_params = len(self.param_names)
        y_range = float(np.ptp(y_cal))
        y_mid = float(np.mean(y_cal))
        x_mid = float(np.mean(x_cal))
        x_range = float(np.ptp(x_cal))

        guess_sets = [
            [1.0] * n_params,
            [y_mid / max(n_params, 1)] * n_params,
            [y_range] * n_params,
        ]
        # A heuristic guess: first param ~ y_min, second ~ y_range,
        # third ~ x_mid, rest ~ 1
        if n_params >= 3:
            smart_guess = [float(np.min(y_cal))] * n_params
            if n_params >= 2:
                smart_guess[1] = y_range
            smart_guess[2] = x_mid if x_mid != 0 else x_range / 2
            for i in range(3, n_params):
                smart_guess[i] = 1.0
            guess_sets.append(smart_guess)

        for p0 in guess_sets:
            try:
                popt, _ = curve_fit(
                    _wrapper, x_cal, y_cal, p0=p0,
                    maxfev=10000, full_output=False,
                )
                result = {name: float(popt[i])
                          for i, name in enumerate(self.param_names)}
                # Sanity check: all finite?
                if all(np.isfinite(v) for v in result.values()):
                    return result
            except Exception:
                continue

        return None

    def compute_data_informed_priors(
        self, x_cal, y_cal
    ) -> Dict[str, Dict]:
        """Return a dict of data-informed prior configs for each parameter.

        Uses least-squares estimates when available; otherwise falls back
        to heuristics based on the X and Y data ranges.

        Returns
        -------
        dict
            Keys are parameter names (including 'sigma'), values are prior
            config dicts like ``{"dist": "Normal", "mu": ..., "sigma": ...}``.
        """
        x_cal = np.asarray(x_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)

        y_range = float(np.ptp(y_cal))
        y_mean = float(np.mean(y_cal))
        y_std = float(np.std(y_cal))
        x_range = float(np.ptp(x_cal))
        x_mean = float(np.mean(x_cal))

        # Scale for "wide but reasonable" prior width
        scale = max(y_range, y_std, 1.0)

        priors: Dict[str, Dict] = {}

        # Try least-squares first
        ls_vals = self._estimate_initvals(x_cal, y_cal)

        for p_name in self.param_names:
            if ls_vals and p_name in ls_vals:
                # Centre the prior on the LS estimate with generous width
                mu = ls_vals[p_name]
                # Width proportional to |estimate| or data scale, whichever
                # is larger — ensures the prior is weakly informative
                sigma = max(abs(mu) * 3.0, scale, 1.0)
                priors[p_name] = {"dist": "Normal",
                                  "mu": round(mu, 4),
                                  "sigma": round(sigma, 2)}
            else:
                # Fallback heuristic: centre on data midpoint, wide spread
                priors[p_name] = {"dist": "Normal",
                                  "mu": round(y_mean, 2),
                                  "sigma": round(scale * 5, 2)}

        # Sigma: HalfNormal with scale based on Y spread
        resid_scale = y_range / 4.0  # rough residual scale
        if ls_vals:
            try:
                ls_pred = self.forward_numpy(ls_vals, x_cal)
                resid_scale = max(float(np.std(y_cal - ls_pred)), 0.1)
            except Exception:
                pass
        priors["sigma"] = {"dist": "HalfNormal",
                           "sigma": round(max(resid_scale * 3, 1.0), 2)}

        return priors

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
