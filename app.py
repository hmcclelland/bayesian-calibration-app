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
from equation_engine import EquationModel, VarianceModel
from app_config import MODE

ALLOW_UPLOAD = (MODE == "local")

# ── Default data from Gelman, Chew & Shnaidman (2004) ────────────────────────
# Standards from Table 2: cockroach allergen Bla g1 ELISA plate
_DEFAULT_X = "0.64\n0.64\n0.32\n0.32\n0.16\n0.16\n0.08\n0.08\n0.04\n0.04\n0.02\n0.02\n0.01\n0.01\n0.001\n0.001"
_DEFAULT_Y = "101.8\n121.4\n105.2\n114.1\n92.7\n93.3\n72.4\n61.1\n57.6\n50.0\n38.5\n35.1\n26.6\n25.0\n14.7\n14.2"
# Unknown 9 measurements for inverse prediction demo
_DEFAULT_Y_NEW = "49.6\n43.8\n24.0\n24.1\n17.3\n17.6\n15.6\n17.1"

# ── Prior distribution choices ────────────────────────────────────────────────
_PRIOR_DISTS = ["Normal", "HalfNormal", "Uniform", "LogNormal",
                "Exponential", "Gamma"]

# Distributions safe to use on the log-scale (unbounded support).
# Positivity-constrained dists (HalfNormal, LogNormal, Gamma, Exponential)
# would cause PyMC to add an internal log-transform on top of the manual
# exp(), leading to a double-transform and inf starting values.
_LOG_SCALE_DISTS = ["Normal", "Uniform"]

# ── Mean model presets ────────────────────────────────────────────────────────
_MEAN_PRESETS = {
    "Custom": "y = b1 + b2 / (1 + (x / b3)**(-b4))",
    "Linear": "y = a + b*x",
    "Quadratic": "y = a + b*x + c*x**2",
    "Saturating exponential": "y = a * (1 - exp(-b*x))",
    "Logistic (4PL)": "y = a + (d - a) / (1 + (x / c)**(-b))",
}

# ── Variance model presets ────────────────────────────────────────────────────
_VARIANCE_PRESETS = {
    "Custom": "sd = (mu / A)**alpha * sigma",
    "Constant": "sd = sigma",
    "Proportional to mean (constant CV)": "sd = mu * sigma",
}

# ── Session-state defaults (must be set before any widget that uses these keys)
if "mean_eq_input" not in st.session_state:
    st.session_state["mean_eq_input"] = _MEAN_PRESETS["Custom"]
if "var_eq_input" not in st.session_state:
    st.session_state["var_eq_input"] = _VARIANCE_PRESETS["Custom"]

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

# ── Callbacks to auto-populate the equation text inputs ───────────────────
def _on_mean_preset_change():
    """When the mean-model preset dropdown changes, update the equation box."""
    chosen = st.session_state["mean_preset"]
    st.session_state["mean_eq_input"] = _MEAN_PRESETS[chosen]

def _on_var_preset_change():
    """When the variance-model preset dropdown changes, update the equation box."""
    chosen = st.session_state["var_preset"]
    st.session_state["var_eq_input"] = _VARIANCE_PRESETS[chosen]

col_mean, col_var = st.columns(2)

with col_mean:
    st.subheader("Mean model")
    mean_preset = st.selectbox(
        "📈 **Preset**",
        list(_MEAN_PRESETS.keys()),
        index=0,  # Custom is default
        help="Choose a common functional form, or select Custom to write your own.",
        key="mean_preset",
        on_change=_on_mean_preset_change,
    )
    # When a preset is chosen, pre-fill the equation; Custom is editable
    equation_input = st.text_input(
        "✏️ **Equation**",
        placeholder="e.g.  y = a + b*x",
        help="Type any equation of the form  y = f(x).  "
             "Letters other than x become parameters to estimate.",
        key="mean_eq_input",
    )

with col_var:
    st.subheader("Variance model")
    var_preset = st.selectbox(
        "📐 **Preset**",
        list(_VARIANCE_PRESETS.keys()),
        index=0,  # Custom is default
        help="Choose how measurement noise scales with the predicted mean, "
             "or select Custom to write your own.",
        key="var_preset",
        on_change=_on_var_preset_change,
    )
    variance_eq_input = st.text_input(
        "✏️ **Variance equation**  (`sd = g(mu, sigma, ...)`)",
        placeholder="e.g.  sd = sigma",
        help="Write the noise standard deviation as a function of mu "
             "(predicted mean) and sigma (base noise scale). Any other "
             "symbol becomes a learnable parameter.",
        key="var_eq_input",
    )

# --- Live parse & LaTeX preview (side-by-side) ----------------------------
eq_model: Optional[EquationModel] = None
var_model: Optional[VarianceModel] = None
variance_model_key = "custom"  # always use the custom variance engine now

if equation_input.strip():
    try:
        eq_model = EquationModel(equation_input)
    except (ValueError, Exception) as exc:
        st.error(f"❌ Mean model error: {exc}")

if variance_eq_input.strip():
    try:
        var_model = VarianceModel(variance_eq_input)
    except (ValueError, Exception) as exc:
        st.error(f"❌ Variance model error: {exc}")

if eq_model is not None:
    col_mean_preview, col_var_preview = st.columns(2)

    with col_mean_preview:
        st.latex(eq_model.latex_str())
        all_param_names = list(eq_model.param_names) + ["σ (noise)"]
        if var_model is not None:
            all_param_names += var_model.param_names
        param_str = ", ".join([f"**{p}**" for p in all_param_names])
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
        if var_model is not None:
            st.latex(var_model.latex_str())
            st.latex(var_model.variance_latex_str())
            if var_model.param_names:
                st.markdown(
                    "Variance parameters: "
                    + ", ".join([f"**{p}**" for p in var_model.param_names])
                )
        else:
            st.info("Enter a valid variance equation above.")

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
                  default_params: Optional[Dict] = None,
                  allowed_dists: Optional[List[str]] = None):
    """Render widgets for a single prior and return a config dict.

    Parameters
    ----------
    allowed_dists : list[str], optional
        If provided, only these distributions are shown in the dropdown.
        Defaults to the full ``_PRIOR_DISTS`` list.
    """
    if default_params is None:
        default_params = {}
    if allowed_dists is None:
        allowed_dists = _PRIOR_DISTS
    # Ensure the default dist is in the allowed list; fall back to first entry
    if default_dist not in allowed_dists:
        default_dist = allowed_dists[0]
    dist = st.selectbox(f"Distribution", allowed_dists,
                        index=allowed_dists.index(default_dist),
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
    elif dist == "Exponential":
        cfg["lam"] = st.number_input("λ (rate)", value=default_params.get("lam", 1.0),
                                     min_value=0.01, format="%.2f",
                                     key=f"{key_prefix}_lam")
    elif dist == "Gamma":
        c1, c2 = st.columns(2)
        with c1:
            cfg["alpha"] = st.number_input("α (shape)", value=default_params.get("alpha", 2.0),
                                           min_value=0.01, format="%.2f",
                                           key=f"{key_prefix}_alpha")
        with c2:
            cfg["beta"] = st.number_input("β (rate)", value=default_params.get("beta", 1.0),
                                          min_value=0.01, format="%.2f",
                                          key=f"{key_prefix}_beta")
    return cfg


# ── Default prior settings per parameter type ─────────────────────────────
_DEFAULT_PRIOR_MAP = {
    "sigma": ("HalfNormal", {"sigma": 10.0}),
}

# ── Compute data-informed priors when both equation and data are available ─
_data_informed_priors: Dict = {}
if eq_model is not None and cal_df is not None:
    try:
        _data_informed_priors = eq_model.compute_data_informed_priors(
            cal_df["X"].values.astype(float),
            cal_df["Y"].values.astype(float),
        )
    except Exception:
        _data_informed_priors = {}


def _get_default_prior(param_name: str, param_source: str):
    """Return (dist_name, params_dict) using data-informed priors when
    available, otherwise sensible generic defaults."""
    if param_name in _data_informed_priors:
        cfg = _data_informed_priors[param_name]
        dist = cfg.get("dist", "Normal")
        params = {k: v for k, v in cfg.items() if k != "dist"}
        return dist, params
    # Fallback generic defaults
    if param_name == "sigma":
        return "HalfNormal", {"sigma": 10.0}
    elif param_source == "variance model":
        return "HalfNormal", {"sigma": 2.0}
    else:
        return "Normal", {"mu": 0.0, "sigma": 10.0}


with st.expander("⚙️ **Advanced Options** — MCMC settings and priors",
                  expanded=False):

    st.markdown("#### MCMC Settings")
    mcmc_c1, mcmc_c2, mcmc_c3, mcmc_c4 = st.columns(4)
    with mcmc_c1:
        iter_sampling = st.number_input("Draws", 200, 5000, 1000, step=200,
                                        key="adv_draws")
    with mcmc_c2:
        chains = st.number_input("Chains", 1, 8, 2, step=1,
                                 key="adv_chains")
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
    st.markdown(
        "Select a parameter from the dropdown to configure its prior. "
        "The list updates automatically when you change the mean or "
        "variance equations."
    )

    prior_config: Dict = {}
    log_scale_params: List[str] = []

    # Build the full list of parameters from both models
    _all_params: List[str] = []
    _param_sources: Dict[str, str] = {}

    if eq_model is not None:
        for p in eq_model.param_names:
            _all_params.append(p)
            _param_sources[p] = "mean model"
        _all_params.append("sigma")
        _param_sources["sigma"] = "noise scale"

    if var_model is not None:
        for p in var_model.param_names:
            if p not in _all_params:
                _all_params.append(p)
                _param_sources[p] = "variance model"

    if _all_params:
        # Parameter picker dropdown
        _display_names = [
            f"{p}  ({_param_sources[p]})" for p in _all_params
        ]
        selected_display = st.selectbox(
            "🎯 **Parameter to configure**",
            _display_names,
            index=0,
            key="prior_param_picker",
        )
        selected_param = _all_params[_display_names.index(selected_display)]

        st.markdown(f"##### Prior for `{selected_param}`")

        # Determine sensible defaults per parameter (data-informed)
        _def_dist, _def_params = _get_default_prior(
            selected_param,
            _param_sources.get(selected_param, ""),
        )

        # Check for stored prior in session state
        _stored_key = f"_prior_cfg_{selected_param}"

        # Log-scale option (only for mean-model params, not sigma)
        use_log = False  # default for sigma / variance-model params
        if selected_param != "sigma" and _param_sources.get(selected_param) == "mean model":
            use_log = st.checkbox(
                "Model on log scale (enforces positivity)",
                value=False,
                key=f"log_{selected_param}",
            )
            if use_log:
                log_scale_params.append(selected_param)
                st.info(
                    f"**Log-scale prior for `{selected_param}`**\n\n"
                    f"The sampler works with `log_{selected_param}` internally "
                    f"and recovers the original parameter via "
                    f"`{selected_param} = exp(log_{selected_param})`.\n\n"
                    f"• Only **Normal** and **Uniform** priors are offered "
                    f"because positivity-constrained distributions "
                    f"(HalfNormal, LogNormal, Gamma, Exponential) would cause "
                    f"PyMC to add a *second* internal log-transform, leading "
                    f"to numerical failures.\n\n"
                    f"• The prior parameters below are specified in "
                    f"**log-space**. For example, Normal(μ=2.30, σ=1.0) in "
                    f"log-space centres the prior at "
                    f"exp(2.30) ≈ 10 in the original scale.",
                    icon="ℹ️",
                )

        # When log-scale is ticked, auto-convert the data-informed
        # defaults from original space → log-space so the user sees
        # sensible starting values without manual calculation.
        if use_log:
            _log_dist = "Normal"
            _orig_mu = _def_params.get("mu", 0.0)
            _orig_sigma = _def_params.get("sigma", 10.0)
            if _orig_mu > 0:
                _log_mu = round(float(np.log(_orig_mu)), 4)
                _log_sigma = round(max(_orig_sigma / _orig_mu, 0.5), 2)
            else:
                # Fallback: use LS estimate if available
                _ls_vals = _data_informed_priors.get(selected_param, {})
                _ls_mu = _ls_vals.get("mu", 1.0)
                if _ls_mu > 0:
                    _log_mu = round(float(np.log(_ls_mu)), 4)
                    _log_sigma = 2.0
                else:
                    _log_mu = 0.0
                    _log_sigma = 3.0
            _def_dist = _log_dist
            _def_params = {"mu": _log_mu, "sigma": _log_sigma}

        cfg = _prior_widget(
            selected_param,
            key_prefix=f"prior_{selected_param}",
            default_dist=_def_dist,
            default_params=_def_params,
            allowed_dists=_LOG_SCALE_DISTS if use_log else None,
        )

        # Store configured prior in session state so it persists
        # when the user switches to another parameter
        st.session_state[_stored_key] = cfg

        # Collect all prior configs (use stored values or data-informed defaults)
        for p in _all_params:
            sk = f"_prior_cfg_{p}"
            if sk in st.session_state:
                prior_config[p] = st.session_state[sk]
            else:
                _fd, _fp = _get_default_prior(
                    p, _param_sources.get(p, ""))
                prior_config[p] = {"dist": _fd, **_fp}

        # Collect log-scale params from session state
        if eq_model is not None:
            for p in eq_model.param_names:
                lk = f"log_{p}"
                if st.session_state.get(lk, False) and p not in log_scale_params:
                    log_scale_params.append(p)

        # Show a summary table of all current priors
        st.markdown("---")
        st.markdown("##### Current prior summary")
        _summary_rows = []
        for p in _all_params:
            pc = prior_config.get(p, {})
            dist_name = pc.get("dist", "Normal")
            if dist_name == "Normal":
                desc = f"Normal(μ={pc.get('mu', 0):.2f}, σ={pc.get('sigma', 10):.2f})"
            elif dist_name == "HalfNormal":
                desc = f"HalfNormal(σ={pc.get('sigma', 10):.2f})"
            elif dist_name == "Uniform":
                desc = f"Uniform({pc.get('lower', 0):.2f}, {pc.get('upper', 1):.2f})"
            elif dist_name == "LogNormal":
                desc = f"LogNormal(μ={pc.get('mu', 0):.2f}, σ={pc.get('sigma', 1):.2f})"
            elif dist_name == "Exponential":
                desc = f"Exponential(λ={pc.get('lam', 1):.2f})"
            elif dist_name == "Gamma":
                desc = f"Gamma(α={pc.get('alpha', 2):.2f}, β={pc.get('beta', 1):.2f})"
            else:
                desc = dist_name
            log_tag = " *(log scale)*" if p in log_scale_params else ""
            _summary_rows.append({
                "Parameter": p,
                "Source": _param_sources.get(p, ""),
                "Prior": desc + log_tag,
            })
        st.dataframe(
            pd.DataFrame(_summary_rows),
            use_container_width=True,
            hide_index=True,
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
                model, sample_initvals = eq_model.build_pymc_model(
                    x_cal, y_cal,
                    prior_config=prior_config,
                    log_scale_params=log_scale_params,
                    variance_model=variance_model_key,
                    variance_eq=var_model,
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
                        initvals=sample_initvals,
                    )
            except Exception as exc:
                st.error(f"\u274c MCMC sampling failed: {exc}")
                st.stop()

        # Extract posterior draws
        posterior: Dict[str, np.ndarray] = {}
        for par in eq_model.param_names:
            posterior[par] = trace.posterior[par].values.flatten()
        posterior["sigma"] = trace.posterior["sigma"].values.flatten()
        # Extract variance model parameters (custom engine)
        if var_model is not None:
            for vp in var_model.param_names:
                if vp not in posterior:
                    posterior[vp] = trace.posterior[vp].values.flatten()

        # Store in session state
        st.session_state["trace"] = trace
        st.session_state["posterior"] = posterior
        st.session_state["eq_model"] = eq_model
        st.session_state["var_model"] = var_model
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
        stored_var_model = st.session_state.get("var_model", None)
        if stored_var_model is not None:
            for vp in stored_var_model.param_names:
                if vp not in summary_vars:
                    summary_vars.append(vp)
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
            # Collect variance-model parameter draws for sd_numpy
            var_param_draws = {}
            if stored_var_model is not None:
                for vp in stored_var_model.param_names:
                    var_param_draws[vp] = posterior.get(vp, np.ones(n_draws))

            inv_rows = []
            all_draws = []

            progress_bar = st.progress(
                0, text="Computing inverse predictions...")
            for yi, y_val in enumerate(y_new_vals_r):
                # Use the custom variance model to compute per-draw noise sd
                if stored_var_model is not None:
                    mu_proxy = np.full(n_draws, y_val)
                    sd_i = stored_var_model.sd_numpy(
                        mu_proxy, sigma_draws, var_param_draws)
                    sd_i = np.maximum(sd_i, 1e-12)
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
