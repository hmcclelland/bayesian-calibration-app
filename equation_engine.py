"""
equation_engine.py
==================
SymPy-powered symbolic equation parser for Bayesian calibration.

Accepts any user-typed equation string like "y = a + b*x" and produces:
  - LaTeX rendering
  - Stan model code (for CmdStanPy MCMC)
  - NumPy-callable forward function
  - Symbolic or numerical inverse function
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# SymPy -> Stan code converter
# ---------------------------------------------------------------------------

def _sympy_to_stan(expr: sp.Expr, x_indexed: str = "x[i]") -> str:
    """Convert a SymPy expression to a valid Stan code string."""
    x_sym = sp.Symbol("x")

    STAN_FUNCS = {
        sp.exp: "exp",
        sp.log: "log",
        sp.sqrt: "sqrt",
        sp.Abs: "fabs",
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",
    }

    def _convert(e):
        if e == x_sym:
            return x_indexed
        if isinstance(e, sp.Symbol):
            return str(e)
        if isinstance(e, sp.Number):
            if isinstance(e, sp.Integer):
                return str(int(e))
            return f"{float(e)}"
        if isinstance(e, sp.Rational) and not isinstance(e, sp.Integer):
            return f"({float(e)})"
        if (isinstance(e, sp.Mul)
                and e.args[0] == sp.Integer(-1)
                and len(e.args) == 2):
            return f"(-{_convert(e.args[1])})"
        if isinstance(e, sp.Pow):
            base = _convert(e.args[0])
            exp_part = _convert(e.args[1])
            return f"pow({base}, {exp_part})"
        if isinstance(e, sp.Function):
            for sym_func, stan_name in STAN_FUNCS.items():
                if isinstance(e, sym_func):
                    args_str = ", ".join(_convert(a) for a in e.args)
                    return f"{stan_name}({args_str})"
            fname = type(e).__name__.lower()
            args_str = ", ".join(_convert(a) for a in e.args)
            return f"{fname}({args_str})"
        if isinstance(e, sp.Add):
            parts = [_convert(a) for a in e.args]
            result = parts[0]
            for p in parts[1:]:
                if p.startswith("(-"):
                    result += f" - {p[2:-1]}"
                elif p.startswith("-"):
                    result += f" - {p[1:]}"
                else:
                    result += f" + {p}"
            return f"({result})"
        if isinstance(e, sp.Mul):
            parts = [_convert(a) for a in e.args]
            return " * ".join(parts)
        return str(e).replace("**", "^")

    return _convert(expr)


# ---------------------------------------------------------------------------
# EquationModel
# ---------------------------------------------------------------------------

class EquationModel:
    """Parse a user-supplied equation string and produce everything needed
    for Bayesian calibration: Stan code, NumPy forward/inverse functions."""

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

    # ---- Stan code generation --------------------------------------------
    def stan_mu_expr(self, x_indexed: str = "x[i]") -> str:
        return _sympy_to_stan(self.rhs_expr, x_indexed=x_indexed)

    def stan_code(self) -> str:
        mu_expr = self.stan_mu_expr("x[i]")
        param_lines = [f"  real {p};" for p in self.param_names]
        param_lines.append("  real<lower=0> sigma;")
        param_block = "\n".join(param_lines)

        prior_lines = [f"  {p} ~ normal(0, 10);" for p in self.param_names]
        prior_lines.append("  sigma ~ exponential(1);")
        prior_block = "\n".join(prior_lines)

        return (
            "data {\n"
            "  int<lower=1> N;\n"
            "  vector[N] x;\n"
            "  vector[N] y;\n"
            "}\n"
            "parameters {\n"
            f"{param_block}\n"
            "}\n"
            "model {\n"
            f"{prior_block}\n"
            "  for (i in 1:N)\n"
            f"    y[i] ~ normal({mu_expr}, sigma);\n"
            "}\n"
            "generated quantities {\n"
            "  vector[N] y_rep;\n"
            "  vector[N] log_lik;\n"
            "  for (i in 1:N) {\n"
            f"    real mu_i = {mu_expr};\n"
            "    y_rep[i] = normal_rng(mu_i, sigma);\n"
            "    log_lik[i] = normal_lpdf(y[i] | mu_i, sigma);\n"
            "  }\n"
            "}\n"
        )

    def __repr__(self):
        return f"EquationModel('{self.equation_str}')  params={self.param_names}"
