"""
Bayesian Calibration & Inverse Prediction — Streamlit App
=========================================================
Run locally:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from typing import Dict, Optional, List
from equation_engine import EquationModel
from app_config import MODE

ALLOW_UPLOAD = (MODE == "local")

# ── Default data from Gelman, Chew & Shnaidman (2004) ────────────────────────
# Standards from Table 2: cockroach allergen Bla g1 ELISA plate
_DEFAULT_X = "0.64\n0.64\n0.32\n0.32\n0.16\n0.16\n0.08\n0.08\n0.04\n0.04\n0.02\n0.02\n0.01\n0.01\n0.001\n0.001"
_DEFAULT_Y = "101.8\n121.4\n105.2\n114.1\n92.7\n93.3\n72.4\n61.1\n57.6\n50.0\n38.5\n35.1\n26.6\n25.0\n14.7\n14.2"
# Unknown 9 measurements for inverse prediction demo
_DEFAULT_Y_NEW = "49.6\n43.8\n24.0\n24.1\n17.3\n17.6\n15.6\n17.1"

# ── Prior distribution choices ────────────────────────────────────────────────
_PRIOR_DISTS = ["Normal", "HalfNormal", "Uniform", "LogNormal"]

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bayesian Calibration Tool",
    page_icon="\U0001f52c",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Header + editable description from description.md
# ══════════════════════════════════════════════════════════════════════════════
st.title("Bayesian Calibration & Inverse Prediction")

_desc_path = pathlib.Path(__file__).parent / "description.md"
if _desc_path.exists():
    st.markdown(_desc_path.read_text())
else:
    st.markdown(
        "Define **any** calibration equation, fit it to data with Bayesian MCMC, "
        "then estimate **X** from new **Y** values with full uncertainty."
    )

# -- "Under the Hood" PDF link ---------------------------------------------
_pdf_static = pathlib.Path(__file__).parent / "static" / "derivation.pdf"
if _pdf_static.exists():
    import os as _os
    _pdf_mtime = int(_os.path.getmtime(_pdf_static))
    st.markdown(
        f'📄 <a href="app/static/derivation.pdf?v={_pdf_mtime}" '
        f'target="_blank"><strong>Under the Hood</strong> — '
        f'Mathematical derivation (PDF)</a>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "📄 *Under the Hood — mathematical derivation PDF not yet compiled. "
        "See `docs/derivation.tex`.*"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Equation syntax help + mode indicator
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("\U0001f4d6 Equation syntax")
    st.markdown("""
**Rules:**
- `x` = independent variable
- `y` = response (left-hand side)
- Everything else = parameter to estimate
- `**` or `^` for powers
- `*` for multiply (required)

**Functions:** `exp()` `log()` `sqrt()`
`sin()` `cos()` `tan()`
`sinh()` `cosh()` `tanh()`

**Examples:**
```
y = a + b*x
y = a + b*x + c*x**2
y = a * x**b
y = a * exp(b*x)
y = a + b*log(x)
y = a / (1 + exp(-b*(x - c)))
y = a * (1 - exp(-b*x))
y = b1 + b2 / (1 + (x/b3)**(-b4))
```
    """)
    if MODE == "local":
        st.divider()
        st.success("\U0001f5a5\ufe0f Running in **local** mode — CSV upload enabled.")
    else:
        st.divider()
        st.info("\u2601\ufe0f Running in **cloud** mode — manual data entry only.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Equation Editor & Variance Model (side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
st.header("Define Your Calibration Model")

# ── Variance model choices ────────────────────────────────────────────────
_VARIANCE_MODELS = {
    "Constant variance": "constant",
    "Proportional to mean (constant CV)": "proportional",
    "Gelman et al. (2004) — learnable exponent": "gelman2004",
}

_VARIANCE_LATEX = {
    "constant": r"\mathrm{Var}(y_i) \;=\; \sigma^2",
    "proportional": r"\mathrm{Var}(y_i) \;=\; \mu_i^{\,2}\,\sigma^2",
    "gelman2004": r"\mathrm{Var}(y_i) \;=\; \left(\frac{\mu_i}{A}\right)^{2\alpha} \sigma^2",
}

_VARIANCE_HELP = {
    "constant": "Noise has the same spread at all signal levels.",
    "proportional": (
        "Standard deviation scales linearly with the predicted mean — "
        "equivalent to assuming a constant coefficient of variation (CV)."
    ),
    "gelman2004": (
        "The exponent α is learned from the data "
        "([Gelman, Chew & Shnaidman 2004](https://doi.org/10.1111/j.0006-341X.2004.00185.x)). "
        "*A* is the geometric mean of the calibration Y values. "
        "α = 0 recovers constant variance; α = 1 ≈ constant CV."
    ),
}

col_mean, col_var = st.columns(2)

with col_mean:
    st.subheader("Mean model")
    st.markdown(
        "Type your equation using Python-style syntax. "
        "Any symbol other than `x` is a parameter to estimate. "
        "See the sidebar for syntax help."
    )
    equation_input = st.text_input(
        "✏️ **Equation**",
        value="y = b1 + b2 / (1 + (x / b3)**(-b4))",
        placeholder="e.g.  y = a + b*x",
        help="Type any equation of the form  y = f(x).  "
             "Letters other than x become parameters to estimate.",
    )

with col_var:
    st.subheader("Variance model")
    st.markdown(
        "Choose how measurement noise scales with the predicted mean response μᵢ = f(xᵢ)."
    )
    variance_label = st.selectbox(
        "📐 **Variance structure**",
        list(_VARIANCE_MODELS.keys()),
        index=0,
        help="Constant = same noise everywhere. "
             "Proportional = noise grows with signal (constant CV). "
             "Gelman 2004 = exponent learned from data.",
    )
    variance_model_key = _VARIANCE_MODELS[variance_label]

# --- Live parse & LaTeX preview (side-by-side) ----------------------------
eq_model: Optional[EquationModel] = None

if equation_input.strip():
    try:
        eq_model = EquationModel(equation_input)

        col_mean_preview, col_var_preview = st.columns(2)

        with col_mean_preview:
            st.latex(eq_model.latex_str())
            param_str = ", ".join(
                [f"**{p}**" for p in eq_model.param_names] + ["**σ** (noise)"]
            )
            if variance_model_key == "gelman2004":
                param_str += ", **α** (variance exponent)"
            st.markdown(f"Parameters to estimate: {param_str}")
            if eq_model.has_symbolic_inverse:
                st.markdown("✅ **Symbolic inverse found:**")
                st.latex(eq_model.inverse_latex_str())
            else:
                st.markdown(
                    "⚠️ No closed-form inverse — will use **numerical root-finding** "
                    "(works fine, just slower for large datasets)."
                )

        with col_var_preview:
            st.latex(_VARIANCE_LATEX[variance_model_key])
            st.markdown(_VARIANCE_HELP[variance_model_key])

    except ValueError as exc:
        st.error(f"❌ {exc}")
    except Exception as exc:
        st.error(f"❌ Unexpected error parsing equation: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Calibration Data
# ══════════════════════════════════════════════════════════════════════════════
st.header("Calibration Data")
st.markdown("Provide paired **(X, Y)** calibration measurements.")

cal_df: Optional[pd.DataFrame] = None

if ALLOW_UPLOAD:
    data_method = st.radio(
        "How would you like to provide calibration data?",
        ["Enter manually", "Upload CSV"],
        horizontal=True,
    )
else:
    data_method = "Enter manually"

if data_method == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload a CSV with columns **X** and **Y** (case-insensitive)",
        type=["csv", "tsv", "txt"],
    )
    if uploaded is not None:
        cal_df = pd.read_csv(uploaded)
        cal_df.columns = [c.strip().upper() for c in cal_df.columns]
        if "X" not in cal_df.columns or "Y" not in cal_df.columns:
            st.error("CSV must contain columns named **X** and **Y**.")
            cal_df = None
else:
    st.markdown("Enter X and Y values (one per line, same length):")
    col1, col2 = st.columns(2)
    with col1:
        x_text = st.text_area("X values", value=_DEFAULT_X, height=200)
    with col2:
        y_text = st.text_area("Y values", value=_DEFAULT_Y, height=200)
    try:
        x_vals = [float(v) for v in x_text.strip().split("\n") if v.strip()]
        y_vals = [float(v) for v in y_text.strip().split("\n") if v.strip()]
        if len(x_vals) == len(y_vals) and len(x_vals) >= 2:
            cal_df = pd.DataFrame({"X": x_vals, "Y": y_vals})
        else:
            st.warning("X and Y must have the same number of values (≥ 2).")
    except ValueError:
        st.error("Could not parse values — make sure each line is a number.")

if cal_df is not None:
    st.subheader("Calibration data preview")
    col_tbl, col_plot = st.columns([1, 2])
    with col_tbl:
        st.dataframe(cal_df, use_container_width=True, height=300)
    with col_plot:
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 3.5))
        ax_scatter.scatter(cal_df["X"], cal_df["Y"], c="black", s=40)
        ax_scatter.set_xlabel("X")
        ax_scatter.set_ylabel("Y")
        ax_scatter.set_title("Calibration data")
        fig_scatter.tight_layout()
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — New Y values
# ══════════════════════════════════════════════════════════════════════════════
st.header("New Y Values for Inverse Prediction")
st.markdown(
    "Enter the **Y** values for which you want to estimate the "
    "corresponding **X**."
)

y_new_vals: Optional[np.ndarray] = None

if ALLOW_UPLOAD:
    new_y_method = st.radio(
        "How would you like to provide new Y values?",
        ["Enter manually", "Upload CSV"],
        horizontal=True,
        key="new_y_radio",
    )
else:
    new_y_method = "Enter manually"

if new_y_method == "Upload CSV":
    uploaded_y = st.file_uploader(
        "Upload a CSV with a column named **Y**",
        type=["csv", "tsv", "txt"],
        key="new_y_upload",
    )
    if uploaded_y is not None:
        df_y = pd.read_csv(uploaded_y)
        df_y.columns = [c.strip().upper() for c in df_y.columns]
        if "Y" in df_y.columns:
            y_new_vals = df_y["Y"].dropna().values.astype(float)
        else:
            st.error("CSV must contain a column named **Y**.")
else:
    y_new_text = st.text_area(
        "Y values (one per line)",
        value=_DEFAULT_Y_NEW,
        height=150,
        key="new_y_text",
    )
    try:
        y_new_vals = np.array(
            [float(v) for v in y_new_text.strip().split("\n") if v.strip()]
        )
    except ValueError:
        st.error("Could not parse — enter one number per line.")

# ══════════════════════════════════════════════════════════════════════════════
# Advanced Options (collapsible)
# ══════════════════════════════════════════════════════════════════════════════


def _prior_widget(label: str, key_prefix: str, default_dist: str = "Normal",
                  default_params: Optional[Dict] = None):
    """Render widgets for a single prior and return a config dict."""
    if default_params is None:
        default_params = {}
    dist = st.selectbox(f"Distribution", _PRIOR_DISTS,
                        index=_PRIOR_DISTS.index(default_dist),
                        key=f"{key_prefix}_dist")
    cfg = {"dist": dist}
    if dist == "Normal":
        c1, c2 = st.columns(2)
        with c1:
            cfg["mu"] = st.number_input("μ", value=default_params.get("mu", 0.0),
                                        format="%.2f", key=f"{key_prefix}_mu")
        with c2:
            cfg["sigma"] = st.number_input("σ", value=default_params.get("sigma", 10.0),
                                           min_value=0.01, format="%.2f",
                                           key=f"{key_prefix}_sigma")
    elif dist == "HalfNormal":
        cfg["sigma"] = st.number_input("σ", value=default_params.get("sigma", 10.0),
                                       min_value=0.01, format="%.2f",
                                       key=f"{key_prefix}_sigma")
    elif dist == "Uniform":
        c1, c2 = st.columns(2)
        with c1:
            cfg["lower"] = st.number_input("Lower", value=default_params.get("lower", 0.0),
                                           format="%.2f", key=f"{key_prefix}_lo")
        with c2:
            cfg["upper"] = st.number_input("Upper", value=default_params.get("upper", 2.0),
                                           format="%.2f", key=f"{key_prefix}_hi")
    elif dist == "LogNormal":
        c1, c2 = st.columns(2)
        with c1:
            cfg["mu"] = st.number_input("μ (log scale)", value=default_params.get("mu", 0.0),
                                        format="%.2f", key=f"{key_prefix}_mu")
        with c2:
            cfg["sigma"] = st.number_input("σ (log scale)",
                                           value=default_params.get("sigma", 1.0),
                                           min_value=0.01, format="%.2f",
                                           key=f"{key_prefix}_sigma")
    return cfg


with st.expander("⚙️ **Advanced Options** — MCMC settings and priors",
                  expanded=False):

    st.markdown("#### MCMC Settings")
    mcmc_c1, mcmc_c2, mcmc_c3, mcmc_c4 = st.columns(4)
    with mcmc_c1:
        chains = st.number_input("Chains", 1, 8, 4, key="adv_chains")
    with mcmc_c2:
        iter_sampling = st.number_input("Draws", 500, 10000, 2000, step=500,
                                        key="adv_draws")
    with mcmc_c3:
        iter_warmup = st.number_input("Warm-up", 200, 5000, 1000, step=200,
                                      key="adv_warmup")
    with mcmc_c4:
        seed = st.number_input("Seed", value=42, key="adv_seed")

    credible_level = st.slider("Credible interval", min_value=0.50,
                               max_value=0.99, value=0.95, step=0.01,
                               key="adv_ci")

    # ── Priors ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Prior Distributions")

    prior_config: Dict = {}
    log_scale_params: List[str] = []

    if eq_model is not None:
        st.markdown("##### Model parameters")
        for p_name in eq_model.param_names:
            with st.container():
                st.markdown(f"**`{p_name}`**")
                use_log = st.checkbox(
                    f"Model on log scale (enforces positivity)",
                    value=False,
                    key=f"log_{p_name}",
                )
                if use_log:
                    log_scale_params.append(p_name)
                    st.caption(f"Prior is on log({p_name}); "
                               f"the model uses {p_name} = exp(log_{p_name}).")
                prior_config[p_name] = _prior_widget(
                    p_name, key_prefix=f"prior_{p_name}",
                    default_dist="Normal",
                    default_params={"mu": 0.0, "sigma": 10.0},
                )

        st.markdown("##### Noise parameters")
        st.markdown("**`sigma`** (σ_y)")
        prior_config["sigma"] = _prior_widget(
            "sigma", key_prefix="prior_sigma",
            default_dist="HalfNormal",
            default_params={"sigma": 10.0},
        )

        if variance_model_key == "gelman2004":
            st.markdown("**`alpha`** (variance exponent)")
            prior_config["alpha"] = _prior_widget(
                "alpha", key_prefix="prior_alpha",
                default_dist="Uniform",
                default_params={"lower": 0.0, "upper": 2.0},
            )
    else:
        st.info("Define an equation above to configure priors.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run
# ══════════════════════════════════════════════════════════════════════════════
st.header("Run Calibration & Inverse Prediction")

run_ready = (
    eq_model is not None
    and cal_df is not None
    and y_new_vals is not None
    and len(y_new_vals) > 0
)

if not run_ready:
    st.info(
        "Complete the steps above (valid equation, calibration data, "
        "and new Y values), then click **Run**."
    )
else:
    if st.button("Run", type="primary", use_container_width=True):

        import pymc as pm
        import arviz as az

        x_cal = cal_df["X"].values.astype(float)
        y_cal = cal_df["Y"].values.astype(float)

        with st.spinner("Building model..."):
            try:
                model = eq_model.build_pymc_model(
                    x_cal, y_cal,
                    prior_config=prior_config,
                    log_scale_params=log_scale_params,
                    variance_model=variance_model_key,
                )
            except Exception as exc:
                st.error(f"\u274c Model build failed: {exc}")
                st.stop()

        with st.spinner("Sampling posterior (this may take a moment)..."):
            try:
                with model:
                    trace = pm.sample(
                        draws=int(iter_sampling),
                        tune=int(iter_warmup),
                        chains=int(chains),
                        random_seed=int(seed),
                        progressbar=False,
                        return_inferencedata=True,
                    )
            except Exception as exc:
                st.error(f"\u274c MCMC sampling failed: {exc}")
                st.stop()

        # Extract posterior draws
        posterior: Dict[str, np.ndarray] = {}
        for par in eq_model.param_names:
            posterior[par] = trace.posterior[par].values.flatten()
        posterior["sigma"] = trace.posterior["sigma"].values.flatten()
        if variance_model_key == "gelman2004":
            posterior["alpha"] = trace.posterior["alpha"].values.flatten()

        # Store in session state
        st.session_state["trace"] = trace
        st.session_state["posterior"] = posterior
        st.session_state["eq_model"] = eq_model
        st.session_state["x_cal"] = x_cal
        st.session_state["y_cal"] = y_cal
        st.session_state["y_new_vals"] = y_new_vals
        st.session_state["variance_model"] = variance_model_key
        st.session_state["run_complete"] = True
        st.session_state["show_inverse"] = False  # reset on new run

    # ══════════════════════════════════════════════════════════════════════
    # Display results
    # ══════════════════════════════════════════════════════════════════════
    if st.session_state.get("run_complete", False):

        import pymc as pm
        import arviz as az

        trace = st.session_state["trace"]
        posterior = st.session_state["posterior"]
        eq_model_r = st.session_state["eq_model"]
        x_cal = st.session_state["x_cal"]
        y_cal = st.session_state["y_cal"]
        y_new_vals_r = st.session_state.get("y_new_vals", y_new_vals)
        stored_variance_model = st.session_state.get("variance_model", "constant")

        # -- MCMC Summary -------------------------------------------------
        st.subheader("MCMC Summary")
        summary_vars = eq_model_r.param_names + ["sigma"]
        if stored_variance_model == "gelman2004":
            summary_vars.append("alpha")
        summary_df = az.summary(trace, var_names=summary_vars)
        st.dataframe(summary_df, use_container_width=True)

        # -- Calibration fit plot ------------------------------------------
        st.subheader("Calibration Fit")
        fig_fit, ax_fit = plt.subplots(figsize=(9, 5))
        ax_fit.scatter(x_cal, y_cal, c="black", zorder=5, s=40, label="Data")

        x_grid = np.linspace(x_cal.min() * 0.9, x_cal.max() * 1.1, 300)
        n_total = len(posterior[eq_model_r.param_names[0]])
        n_curves = min(300, n_total)
        idx_curves = np.random.choice(n_total, n_curves, replace=False)
        for i in idx_curves:
            p_i = {k: posterior[k][i] for k in eq_model_r.param_names}
            yg = eq_model_r.forward_numpy(p_i, x_grid)
            ax_fit.plot(x_grid, yg, alpha=0.02, color="steelblue")

        ax_fit.set_xlabel("X")
        ax_fit.set_ylabel("Y")
        ax_fit.set_title("Forward model fit with posterior uncertainty")
        ax_fit.legend()
        fig_fit.tight_layout()
        st.pyplot(fig_fit)
        plt.close(fig_fit)

        # ══════════════════════════════════════════════════════════════════
        # Residual Diagnostics
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Residual Diagnostics")

        median_params = {p: np.median(posterior[p])
                         for p in eq_model_r.param_names}
        y_pred = eq_model_r.forward_numpy(median_params, x_cal)
        residuals = y_cal - y_pred

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            fig_rvf, ax_rvf = plt.subplots(figsize=(5, 3.5))
            ax_rvf.scatter(y_pred, residuals, c="steelblue", s=40,
                           edgecolors="k", linewidths=0.5)
            ax_rvf.axhline(0, color="red", ls="--", lw=1)
            ax_rvf.set_xlabel("Fitted values (Ŷ)")
            ax_rvf.set_ylabel("Residuals (Y − Ŷ)")
            ax_rvf.set_title("Residuals vs Fitted")
            fig_rvf.tight_layout()
            st.pyplot(fig_rvf)
            plt.close(fig_rvf)

        with col_res2:
            fig_rvx, ax_rvx = plt.subplots(figsize=(5, 3.5))
            ax_rvx.scatter(x_cal, residuals, c="steelblue", s=40,
                           edgecolors="k", linewidths=0.5)
            ax_rvx.axhline(0, color="red", ls="--", lw=1)
            ax_rvx.set_xlabel("X")
            ax_rvx.set_ylabel("Residuals (Y − Ŷ)")
            ax_rvx.set_title("Residuals vs X")
            fig_rvx.tight_layout()
            st.pyplot(fig_rvx)
            plt.close(fig_rvx)

        # -- Statistical tests ---------------------------------------------
        st.markdown("**Statistical tests for residual structure:**")

        from statsmodels.stats.diagnostic import het_breuschpagan
        from scipy.stats import norm as _norm
        import statsmodels.api as sm

        exog_bp = sm.add_constant(y_pred)
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, exog_bp)

        def _wald_wolfowitz_runs_test(residuals_arr):
            signs = np.array(residuals_arr) > 0
            n_pos = int(signs.sum())
            n_neg = int((~signs).sum())
            n = n_pos + n_neg
            if n_pos == 0 or n_neg == 0 or n < 3:
                return np.nan, np.nan
            runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
            e_runs = 1.0 + (2.0 * n_pos * n_neg) / n
            var_runs = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (
                n**2 * (n - 1.0))
            if var_runs <= 0:
                return np.nan, np.nan
            z = (runs - e_runs) / np.sqrt(var_runs)
            p_value = 2.0 * _norm.sf(np.abs(z))
            return z, p_value

        runs_stat, runs_p = _wald_wolfowitz_runs_test(residuals)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Breusch–Pagan test (heteroscedasticity)")
            st.markdown(
                f"- LM statistic: **{bp_lm:.4f}**\n"
                f"- *p*-value: **{bp_lm_p:.4f}**"
            )
            if bp_lm_p < 0.05:
                st.warning(
                    "⚠️ Significant heteroscedasticity detected (*p* < 0.05). "
                    "The variance of the residuals is not constant across "
                    "fitted values."
                )
            else:
                st.success(
                    "✅ No significant heteroscedasticity (*p* ≥ 0.05)."
                )

        with col_t2:
            st.markdown("##### Wald–Wolfowitz runs test (randomness)")
            st.markdown(
                f"- Test statistic: **{runs_stat:.4f}**\n"
                f"- *p*-value: **{runs_p:.4f}**"
            )
            if runs_p < 0.05:
                st.warning(
                    "⚠️ Significant non-randomness in residuals (*p* < 0.05). "
                    "There may be systematic structure the model does not "
                    "capture."
                )
            else:
                st.success("✅ Residuals appear random (*p* ≥ 0.05).")

        # -- Guided diagnostic questions -----------------------------------
        st.markdown("---")
        st.markdown("#### 🔍 Interpreting the residual plots")
        st.markdown(
            "Use the plots and test results above to check whether your "
            "model is a good fit. Here are some questions to guide you:"
        )

        with st.expander(
            "**Can you see a pattern (curve, trend) in the residuals?**",
            expanded=False,
        ):
            st.markdown(
                "If the residuals show a systematic curve or trend rather "
                "than a random scatter around zero, the current equation "
                "may not capture the true relationship between X and Y.\n\n"
                "**What to try:**\n"
                "- Think about the **physical or biological process** "
                "underlying your assay. Does the relationship between X "
                "and Y have a known functional form? For example, enzyme "
                "kinetics often follow a Michaelis–Menten curve, "
                "fluorescence assays may saturate exponentially, and "
                "dose-response curves are typically sigmoidal. Choose a "
                "model that reflects the mechanism, not just one that "
                "fits the numbers.\n"
                "- If you're unsure of the underlying mechanism, look at "
                "the scatter plot of your calibration data for visual "
                "clues about the shape — is it concave, convex, "
                "S-shaped?\n"
                "- Avoid blindly adding polynomial terms to improve the "
                "fit. A high-order polynomial may follow the calibration "
                "data closely but will extrapolate poorly and give "
                "unreliable inverse predictions outside the calibration "
                "range."
            )

        with st.expander(
            "**Does the spread of residuals change systematically "
            "(bigger at one end)?**",
            expanded=False,
        ):
            st.markdown(
                "If the residuals fan out — e.g. small residuals at low "
                "X (or Ŷ) and large residuals at high X — the noise may "
                "not be constant. The Breusch–Pagan test above checks "
                "for this formally.\n\n"
                "**A note on the test:** The Breusch–Pagan test is only "
                "an indicator. With small calibration datasets it has "
                "limited power, and with very large datasets it can flag "
                "trivially small effects. **Look at the residual plots "
                "first.** If you can't see an obvious fan or funnel "
                "shape in the residuals, the constant-variance "
                "assumption is probably fine for your purposes — even if "
                "the test returns a low *p*-value.\n\n"
                "**If the spread clearly changes:**\n"
                "- Change the **Variance model** dropdown (Step 1) from "
                "*Constant* to *Proportional to mean* (constant CV) or "
                "*Gelman et al. (2004)* to let the model learn how "
                "noise scales with the mean.\n"
                "- Alternatively, **log-transform your data** before "
                "fitting. Replace your Y values with `log(Y)` and fit "
                "the model to the transformed data.\n"
                "- If your assay has a known coefficient-of-variation "
                "(CV), the *Proportional to mean* option or working in "
                "log-space naturally accounts for constant-CV noise."
            )

        with st.expander(
            "**Are there one or two outliers far from zero?**",
            expanded=False,
        ):
            st.markdown(
                "Isolated large residuals may indicate data entry errors, "
                "sample preparation problems, or genuine outliers.\n\n"
                "**What to try:**\n"
                "- Double-check the raw data for transcription errors.\n"
                "- If the outlier is real, consider whether that standard "
                "should be excluded or whether a more robust model is "
                "needed.\n"
                "- As a quick sanity check, remove the suspect point from "
                "your calibration data and re-run — if the fit improves "
                "dramatically, that point was likely problematic."
            )

        with st.expander(
            "**Do the residuals look random and centred around zero?**",
            expanded=False,
        ):
            st.markdown(
                "Great — this is what you want to see! If both "
                "statistical tests also pass (*p* ≥ 0.05), the model "
                "assumptions appear to be satisfied and you can be "
                "confident in the inverse predictions below."
            )

        # ══════════════════════════════════════════════════════════════════
        # Inverse Prediction (hidden until button click)
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        if st.button("Show inverse predictions", type="primary",
                     use_container_width=True, key="show_inverse_btn"):
            st.session_state["show_inverse"] = True

        if st.session_state.get("show_inverse", False):

            st.subheader("Inverse Predictions")
            alpha_tail = (1 - credible_level) / 2
            sigma_draws = posterior["sigma"]
            n_draws = len(sigma_draws)
            x_hint = float(x_cal.mean())
            x_lo_range = float(x_cal.min()) - 3 * float(np.ptp(x_cal))
            x_hi_range = float(x_cal.max()) + 3 * float(np.ptp(x_cal))

            # Compute per-draw noise sd for the new Y values
            # For heteroscedastic models, noise depends on the predicted
            # mean at the (unknown) x*, so we approximate using the
            # observed y* as a stand-in for μ.
            if stored_variance_model == "gelman2004":
                alpha_draws = posterior["alpha"]
                y_cal_pos = np.maximum(y_cal, 1e-12)
                A_val = float(np.exp(np.mean(np.log(y_cal_pos))))
            else:
                alpha_draws = None
                A_val = None

            inv_rows = []
            all_draws = []

            progress_bar = st.progress(
                0, text="Computing inverse predictions...")
            for yi, y_val in enumerate(y_new_vals_r):
                if stored_variance_model == "gelman2004":
                    # sd_i = |y_val / A|^alpha * sigma
                    sd_i = np.abs(y_val / A_val) ** alpha_draws * sigma_draws
                elif stored_variance_model == "proportional":
                    # sd_i = |y_val| * sigma  (constant CV)
                    sd_i = np.abs(y_val) * sigma_draws
                else:
                    sd_i = sigma_draws
                y_star = y_val + np.random.randn(n_draws) * sd_i
                x_draws = eq_model_r.inverse_numpy(
                    y_star, posterior,
                    x_hint=x_hint,
                    x_range=(x_lo_range, x_hi_range),
                )
                x_draws = np.asarray(x_draws, dtype=float)
                x_draws = x_draws[np.isfinite(x_draws)]
                all_draws.append(x_draws)

                if len(x_draws) > 0:
                    lo = np.percentile(x_draws, 100 * alpha_tail)
                    hi = np.percentile(x_draws, 100 * (1 - alpha_tail))
                    inv_rows.append({
                        "Y_observed": y_val,
                        "X_median": np.median(x_draws),
                        "X_mean": np.mean(x_draws),
                        "X_sd": np.std(x_draws),
                        f"X_lo ({credible_level:.0%})": lo,
                        f"X_hi ({credible_level:.0%})": hi,
                    })
                else:
                    inv_rows.append({
                        "Y_observed": y_val,
                        "X_median": np.nan,
                        "X_mean": np.nan,
                        "X_sd": np.nan,
                        f"X_lo ({credible_level:.0%})": np.nan,
                        f"X_hi ({credible_level:.0%})": np.nan,
                    })
                progress_bar.progress(
                    (yi + 1) / len(y_new_vals_r),
                    text=f"Inverse prediction {yi + 1}/{len(y_new_vals_r)}",
                )
            progress_bar.empty()

            df_result = pd.DataFrame(inv_rows)
            st.dataframe(df_result, use_container_width=True)

            csv_buf = df_result.to_csv(index=False)
            st.download_button(
                "Download results as CSV",
                csv_buf,
                file_name="inverse_predictions.csv",
                mime="text/csv",
            )

            # -- Posterior histograms --------------------------------------
            st.subheader("Posterior Distributions of X")
            n_new = len(y_new_vals_r)
            n_cols = min(n_new, 3)
            n_rows_fig = int(np.ceil(n_new / n_cols))
            fig_inv, axes_inv = plt.subplots(
                n_rows_fig, n_cols,
                figsize=(5 * n_cols, 4 * n_rows_fig),
                squeeze=False,
            )
            for idx_y, (y_val, x_draws) in enumerate(
                zip(y_new_vals_r, all_draws)
            ):
                ax = axes_inv[idx_y // n_cols][idx_y % n_cols]
                if len(x_draws) == 0:
                    ax.text(
                        0.5, 0.5, "No valid\ndraws",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="red",
                    )
                else:
                    ax.hist(x_draws, bins=60, density=True,
                            color="steelblue", alpha=0.7)
                    lo = np.percentile(x_draws, 100 * alpha_tail)
                    hi = np.percentile(x_draws, 100 * (1 - alpha_tail))
                    ax.axvline(lo, color="red", ls="--",
                               label=f"{credible_level:.0%} CI")
                    ax.axvline(hi, color="red", ls="--")
                    ax.axvline(np.median(x_draws), color="orange",
                               label="Median")
                    ax.legend(fontsize=8)
                ax.set_title(f"Y = {y_val:.4g}")
                ax.set_xlabel("X")

            for idx_y in range(n_new, n_rows_fig * n_cols):
                axes_inv[idx_y // n_cols][idx_y % n_cols].set_visible(False)

            fig_inv.suptitle(
                "Inverse-prediction posterior distributions", fontsize=14)
            fig_inv.tight_layout()
            st.pyplot(fig_inv)
            plt.close(fig_inv)

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<center><small>Bayesian Calibration Tool — Powered by "
    "<a href='https://www.pymc.io/'>PyMC</a>, "
    "<a href='https://www.sympy.org/'>SymPy</a> & "
    "<a href='https://streamlit.io/'>Streamlit</a></small></center>",
    unsafe_allow_html=True,
)
