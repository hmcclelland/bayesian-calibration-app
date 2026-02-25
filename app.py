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
from app_config import MODE, GA_TRACKING_ID

ALLOW_UPLOAD = (MODE == "local")

# ── Default data from Gelman, Chew & Shnaidman (2004) ────────────────────────
# Standards from Table 2: cockroach allergen Bla g1 ELISA plate
_DEFAULT_X = "0.64\n0.64\n0.32\n0.32\n0.16\n0.16\n0.08\n0.08\n0.04\n0.04\n0.02\n0.02\n0.01\n0.01\n0.001\n0.001"
_DEFAULT_Y = "101.8\n121.4\n105.2\n114.1\n92.7\n93.3\n72.4\n61.1\n57.6\n50.0\n38.5\n35.1\n26.6\n25.0\n14.7\n14.2"
# Unknown 9 measurements for inverse prediction demo
_DEFAULT_Y_NEW = "49.6\n43.8\n24.0\n24.1\n17.3\n17.6\n15.6\n17.1\n70\n80\n90"

# ── Example 1 — synthetic linear data (fixed seed for reproducibility) ────────
_rng_ex1 = np.random.default_rng(42)
_EX1_X_ARR = np.round(_rng_ex1.uniform(0, 50, 20), 2)
_EX1_Y_ARR = np.round(10 + 5 * _EX1_X_ARR + _rng_ex1.normal(0, 5, 20), 2)
_EX1_X_STR = "\n".join(str(v) for v in _EX1_X_ARR)
_EX1_Y_STR = "\n".join(str(v) for v in _EX1_Y_ARR)
_EX1_Y_NEW_STR = "100\n200"

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
    "Custom": "sd = (mu / [A])**alpha * sigma",
    "Constant": "sd = sigma",
    "Proportional to mean (constant CV)": "sd = mu * sigma",
}

# ── Session-state defaults (must be set before any widget that uses these keys)
if "mean_eq_input" not in st.session_state:
    st.session_state["mean_eq_input"] = _MEAN_PRESETS["Linear"]
if "var_eq_input" not in st.session_state:
    st.session_state["var_eq_input"] = _VARIANCE_PRESETS["Constant"]
if "cal_x_text" not in st.session_state:
    st.session_state["cal_x_text"] = ""
if "cal_y_text" not in st.session_state:
    st.session_state["cal_y_text"] = ""
if "new_y_text" not in st.session_state:
    st.session_state["new_y_text"] = ""

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bayesian Calibration Tool",
    page_icon=None,
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Google Analytics consent banner
# Disable entirely by setting GA_TRACKING_ID = "" in app_config.py
# ══════════════════════════════════════════════════════════════════════════════
if GA_TRACKING_ID:
    if "ga_consent" not in st.session_state:
        st.session_state["ga_consent"] = None  # None = not yet decided

    if st.session_state["ga_consent"] is None:
        banner = st.container()
        with banner:
            st.markdown(
                "<div style='background:#f0f2f6;padding:10px 16px;"
                "border-radius:6px;margin-bottom:8px'>"
                "This site uses cookies to understand how it is used "
                "(Google Analytics). No personally identifiable information "
                "is collected.</div>",
                unsafe_allow_html=True,
            )
            _c1, _c2, _c3 = st.columns([1, 1, 6])
            with _c1:
                if st.button("Accept", type="primary",
                             use_container_width=True, key="_ga_accept"):
                    st.session_state["ga_consent"] = True
                    st.rerun()
            with _c2:
                if st.button("Decline", use_container_width=True,
                             key="_ga_decline"):
                    st.session_state["ga_consent"] = False
                    st.rerun()

    if st.session_state.get("ga_consent") is True:
        import streamlit.components.v1 as _components
        _components.html(
            f"""
            <script async
              src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}">
            </script>
            <script>
              window.dataLayer = window.dataLayer || [];
              function gtag(){{dataLayer.push(arguments);}}
              gtag('js', new Date());
              gtag('config', '{GA_TRACKING_ID}');
            </script>
            """,
            height=0,
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
        f'<a href="app/static/derivation.pdf?v={_pdf_mtime}" '
        f'target="_blank"><strong>Under the Hood</strong> — '
        f'Mathematical derivation (PDF)</a>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "*Under the Hood — mathematical derivation PDF not yet compiled. "
        "See `docs/derivation.tex`.*"
    )

if MODE != "local":
    st.info(
        "**Online version:** data entry is by manual text input only. "
        "To upload CSV files, download and run the app locally — "
        "[get it on GitHub](YOUR_GITHUB_URL_HERE)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Equation syntax help + mode indicator
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Equation syntax")
    st.markdown("""
**Rules:**
- `x` = independent variable
- `y` = response (left-hand side)
- Everything else = parameter to estimate
- `**` or `^` for powers
- `*` for multiply (required)
- Wrap a parameter name in square brackets
  (e.g. `[A]`) to make it a **prescribed constant**
  — it won't be fitted; you supply its value directly.

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

**Variance model:**
- `mu` = predicted mean from the calibration model
- `sigma` = base noise scale (always available)
- `x` = the calibration X value (available in variance equations)
- Wrap in `[brackets]` to make a **prescribed constant**

```
sd = (mu / [A])**alpha * sigma
sd = sigma * x**0.5
sd = sigma * (1 + alpha * x)
```
Here `[A]` is prescribed (you set its value),
while `alpha` and `sigma` are estimated.
    """)
    if MODE == "local":
        st.divider()
        st.success("Running in **local** mode — CSV upload enabled.")
    else:
        st.divider()
        st.info("Running in **cloud** mode — manual data entry only.")

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
        "**Preset**",
        list(_MEAN_PRESETS.keys()),
        index=list(_MEAN_PRESETS.keys()).index("Linear"),
        help="Choose a common functional form, or select Custom to write your own.",
        key="mean_preset",
        on_change=_on_mean_preset_change,
    )
    # When a preset is chosen, pre-fill the equation; Custom is editable
    equation_input = st.text_input(
        "**Equation**",
        placeholder="Enter mean model equation here (e.g. y = a + b*x)",
        help="Type any equation of the form  y = f(x).  "
             "Letters other than x become parameters to estimate.",
        key="mean_eq_input",
    )

with col_var:
    st.subheader("Variance model")
    var_preset = st.selectbox(
        "**Preset**",
        list(_VARIANCE_PRESETS.keys()),
        index=list(_VARIANCE_PRESETS.keys()).index("Constant"),
        help="Choose how measurement noise scales with the predicted mean, "
             "or select Custom to write your own.",
        key="var_preset",
        on_change=_on_var_preset_change,
    )
    variance_eq_input = st.text_input(
        "**Variance equation**  (`sd = g(mu, sigma, ...)`)",
        placeholder="e.g.  sd = sigma",
        help="Write the noise standard deviation as a function of mu "
             "(predicted mean) and sigma (base noise scale). Any other "
             "symbol becomes a learnable parameter. Wrap a symbol name "
             "in square brackets (e.g. [A]) to make it a prescribed constant.",
        key="var_eq_input",
    )
    st.caption("`mu` = the predicted mean of the calibration model at each data point. `x` is also available.")

# --- Live parse & LaTeX preview (side-by-side) ----------------------------
eq_model: Optional[EquationModel] = None
var_model: Optional[VarianceModel] = None
variance_model_key = "custom"  # always use the custom variance engine now

if equation_input.strip():
    try:
        eq_model = EquationModel(equation_input)
    except (ValueError, Exception) as exc:
        st.error(f"Mean model error: {exc}")

if variance_eq_input.strip():
    try:
        var_model = VarianceModel(variance_eq_input)
    except (ValueError, Exception) as exc:
        st.error(f"Variance model error: {exc}")

if eq_model is not None:
    col_mean_preview, col_var_preview = st.columns(2)

    with col_mean_preview:
        st.latex(eq_model.latex_str())
        all_param_names = list(eq_model.param_names) + ["σ (noise)"]
        if var_model is not None:
            all_param_names += var_model.param_names
        param_str = ", ".join([f"**{p}**" for p in all_param_names])
        st.markdown(f"Parameters to estimate: {param_str}")

    with col_var_preview:
        if var_model is not None:
            st.latex(var_model.latex_str())
            st.latex(var_model.variance_latex_str())
            if var_model.param_names:
                st.markdown(
                    "Variance parameters: "
                    + ", ".join([f"**{p}**" for p in var_model.param_names])
                )
            if var_model.prescribed_names:
                st.markdown(
                    "Prescribed constants: "
                    + ", ".join([f"**{p}**" for p in var_model.prescribed_names])
                )
        else:
            st.info("Enter a valid variance equation above.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Calibration Data
# ══════════════════════════════════════════════════════════════════════════════

# ── Callbacks for example / clear buttons ────────────────────────────────────
def _load_example1():
    st.session_state["mean_eq_input"] = _MEAN_PRESETS["Linear"]
    st.session_state["mean_preset"] = "Linear"
    st.session_state["var_eq_input"] = _VARIANCE_PRESETS["Constant"]
    st.session_state["var_preset"] = "Constant"
    st.session_state["cal_x_text"] = _EX1_X_STR
    st.session_state["cal_y_text"] = _EX1_Y_STR
    st.session_state["new_y_text"] = _EX1_Y_NEW_STR

def _load_example2():
    st.session_state["mean_eq_input"] = _MEAN_PRESETS["Custom"]
    st.session_state["mean_preset"] = "Custom"
    st.session_state["var_eq_input"] = _VARIANCE_PRESETS["Custom"]
    st.session_state["var_preset"] = "Custom"
    st.session_state["cal_x_text"] = _DEFAULT_X
    st.session_state["cal_y_text"] = _DEFAULT_Y
    st.session_state["new_y_text"] = _DEFAULT_Y_NEW

def _clear_all():
    st.session_state["mean_eq_input"] = _MEAN_PRESETS["Linear"]
    st.session_state["mean_preset"] = "Linear"
    st.session_state["var_eq_input"] = _VARIANCE_PRESETS["Constant"]
    st.session_state["var_preset"] = "Constant"
    st.session_state["cal_x_text"] = ""
    st.session_state["cal_y_text"] = ""
    st.session_state["new_y_text"] = ""

st.markdown(
    "<style>div[data-testid='stVerticalBlockBorderWrapper']"
    "{background-color:#f0f2f6;}</style>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    st.caption("Load an example or start fresh:")
    _btn_col1, _btn_col2, _btn_col3 = st.columns(3)
    with _btn_col1:
        st.button("Example 1 — linear data", on_click=_load_example1,
                  use_container_width=True,
                  help="Load synthetic linear calibration data (y = 10 + 5x + noise)")
    with _btn_col2:
        st.button("Example 2 — Gelman 2004 ELISA", on_click=_load_example2,
                  use_container_width=True,
                  help="Load the cockroach allergen ELISA data from Gelman et al. (2004)")
    with _btn_col3:
        st.button("Clear all / enter my own data", on_click=_clear_all,
                  use_container_width=True,
                  help="Reset all fields and start fresh")

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
        x_text = st.text_area("X values", height=200, key="cal_x_text",
                              placeholder="Enter x calibration data here\n(one value per line)")
    with col2:
        y_text = st.text_area("Y values", height=200, key="cal_y_text",
                              placeholder="Enter y calibration data here\n(one value per line)")
    try:
        x_vals = [float(v) for v in x_text.strip().split("\n") if v.strip()]
        y_vals = [float(v) for v in y_text.strip().split("\n") if v.strip()]
        if len(x_vals) == len(y_vals) and len(x_vals) >= 2:
            cal_df = pd.DataFrame({"X": x_vals, "Y": y_vals})
        elif len(x_vals) > 0 or len(y_vals) > 0:
            st.warning("X and Y must have the same number of values (≥ 2).")
    except ValueError:
        st.error("Could not parse values — make sure each line is a number.")

if cal_df is not None:
    st.subheader("Calibration data preview")
    fig_scatter, ax_scatter = plt.subplots(figsize=(9, 4))
    ax_scatter.scatter(cal_df["X"], cal_df["Y"], c="black", s=40)
    ax_scatter.set_xlabel("X")
    ax_scatter.set_ylabel("Y")
    ax_scatter.set_title("Calibration data")
    fig_scatter.tight_layout()
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)
    st.download_button(
        "Download calibration data CSV",
        cal_df.to_csv(index=False),
        file_name="calibration_data.csv",
        mime="text/csv",
    )

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
        height=150,
        key="new_y_text",
        placeholder="Enter new y values here\n(one per line)",
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

# prescribed_values must be initialised BEFORE the expander so it is
# visible in the outer scope (results display, model build, etc.).
prescribed_values: Dict[str, float] = {}

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

# ── Compute default prescribed values when data is available ──────────────
_prescribed_defaults: Dict[str, float] = {}
if var_model is not None and cal_df is not None:
    y_arr = cal_df["Y"].values.astype(float)
    for pname in var_model.prescribed_names:
        # Default: A → mean(y).  Generic fallback for other names: mean(y).
        _prescribed_defaults[pname] = float(np.mean(y_arr))


def _get_default_prior(param_name: str, param_source: str):
    """Return (dist_name, params_dict) — simple, non-data-informed defaults."""
    # Fallback generic defaults
    if param_name == "sigma":
        # Default: Uniform(0, 50) — wide positive (uniform) prior
        return "Uniform", {"lower": 0.0, "upper": 50.0}
    elif param_name == "alpha":
        return "Uniform", {"lower": 0.0, "upper": 2.0}
    elif param_source == "variance model":
        return "Uniform", {"lower": 0.0, "upper": 2.0}
    else:
        # Mean model params (b1, b2, b3, b4, etc.) → N(0, 100)
        return "Normal", {"mu": 0.0, "sigma": 100.0}


with st.expander("**Advanced Options** — MCMC settings and priors",
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

    # ── Prescribed parameters (fixed constants) ──────────────────────────
    if var_model is not None and var_model.prescribed_names:
        st.markdown("---")
        st.markdown("#### Prescribed Constants")
        st.markdown(
            "These parameters (wrapped in `[brackets]` in the variance equation) "
            "are **not fitted** — set their values here."
        )
        for pname in var_model.prescribed_names:
            default_val = _prescribed_defaults.get(pname, 1.0)
            prescribed_values[pname] = st.number_input(
                f"**{pname}** (prescribed)",
                value=default_val,
                format="%.4f",
                key=f"prescribed_{pname}",
                help=f"Fixed value for {pname}. Default is mean(Y) from "
                     f"calibration data.",
            )

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

    # Persistent set for log-scale choices — survives when the checkbox
    # widget for a different parameter is not rendered (Streamlit removes
    # un-rendered widget keys from session state at end of each run).
    if "_log_scale_set" not in st.session_state:
        st.session_state["_log_scale_set"] = set()

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
            "**Parameter to configure**",
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
                value=selected_param in st.session_state["_log_scale_set"],
                key=f"log_{selected_param}",
            )
            # Sync the persistent set with the checkbox value
            if use_log:
                st.session_state["_log_scale_set"].add(selected_param)
            else:
                st.session_state["_log_scale_set"].discard(selected_param)

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
                )

        # When log-scale is ticked, auto-convert the defaults
        # from original space → log-space so the user sees
        # sensible starting values without manual calculation.
        if use_log:
            _log_dist = "Normal"
            _orig_mu = _def_params.get("mu", 0.0)
            _orig_sigma = _def_params.get("sigma", 10.0)
            if _orig_mu > 0:
                _log_mu = round(float(np.log(_orig_mu)), 4)
                _log_sigma = round(max(_orig_sigma / _orig_mu, 0.5), 2)
            else:
                # Fallback: vague Normal on log-space
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

        # Collect log-scale params from the persistent set (not widget keys)
        if eq_model is not None:
            for p in eq_model.param_names:
                if p in st.session_state["_log_scale_set"] and p not in log_scale_params:
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
                    prescribed_values=prescribed_values,
                )
            except Exception as exc:
                st.error(f"Model build failed: {exc}")
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
                st.error(f"MCMC sampling failed: {exc}")
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
        st.session_state["prescribed_values"] = prescribed_values
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
        prescribed_values = st.session_state.get("prescribed_values", {})

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
        st.download_button(
            "Download MCMC summary CSV",
            summary_df.to_csv(),
            file_name="mcmc_summary.csv",
            mime="text/csv",
        )

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

        # -- Compute posterior prediction envelope on a fine x-grid --------
        x_grid_res = np.linspace(x_cal.min() * 0.9, x_cal.max() * 1.1, 200)
        n_total_res = len(posterior[eq_model_r.param_names[0]])
        n_env = min(500, n_total_res)
        idx_env = np.random.choice(n_total_res, n_env, replace=False)

        # For each posterior draw, compute predicted mean and predicted sd
        # at every grid point.  Prediction bands are computed analytically
        # (mean_shift ± k·sd) rather than by MC sampling, so they are smooth.
        mu_grid_draws = np.empty((n_env, len(x_grid_res)))
        sd_grid_draws = np.empty((n_env, len(x_grid_res)))
        sigma_draws_arr = posterior["sigma"]

        # Collect variance-model parameter draws
        _var_param_draws_res = {}
        if stored_var_model is not None:
            for vp in stored_var_model.param_names:
                _var_param_draws_res[vp] = posterior.get(
                    vp, np.ones(n_total_res))

        for j, idx in enumerate(idx_env):
            p_j = {k: posterior[k][idx] for k in eq_model_r.param_names}
            mu_j = eq_model_r.forward_numpy(p_j, x_grid_res)
            mu_grid_draws[j] = mu_j
            # Compute observation-level sd at each grid point
            if stored_var_model is not None:
                vp_j = {vp: _var_param_draws_res[vp][idx]
                        for vp in stored_var_model.param_names}
                sd_j = stored_var_model.sd_numpy(
                    mu_j,
                    np.full_like(mu_j, sigma_draws_arr[idx]),
                    {k: np.full_like(mu_j, v) for k, v in vp_j.items()},
                    prescribed_params=prescribed_values,
                    **({'x': x_grid_res}
                       if getattr(stored_var_model, 'uses_x', False) else {}),
                )
            else:
                sd_j = np.full_like(mu_j, sigma_draws_arr[idx])
            sd_grid_draws[j] = np.maximum(sd_j, 1e-12)

        # Posterior median of the mean curve (for the "zero" residual line)
        mu_grid_median = np.median(mu_grid_draws, axis=0)

        # Smooth prediction bands: for each draw, the ±1σ / ±2σ boundaries
        # are deterministic smooth functions of x.  Taking percentiles of
        # these smooth curves across draws gives smooth shaded regions.
        mean_shift = mu_grid_draws - mu_grid_median[np.newaxis, :]

        pct_1s_lo = np.percentile(mean_shift - sd_grid_draws,   15.87, axis=0)
        pct_1s_hi = np.percentile(mean_shift + sd_grid_draws,   84.13, axis=0)
        pct_2s_lo = np.percentile(mean_shift - 2*sd_grid_draws,  2.28, axis=0)
        pct_2s_hi = np.percentile(mean_shift + 2*sd_grid_draws, 97.72, axis=0)

        # -- Compute observation-level predicted sd at each calibration x
        #    using posterior medians (for standardised residuals) -----------
        median_sigma = np.median(sigma_draws_arr)
        if stored_var_model is not None:
            median_var_params = {
                vp: np.median(_var_param_draws_res[vp])
                for vp in stored_var_model.param_names
            }
            sd_at_cal = stored_var_model.sd_numpy(
                y_pred,
                np.full_like(y_pred, median_sigma),
                {k: np.full_like(y_pred, v)
                 for k, v in median_var_params.items()},
                prescribed_params=prescribed_values,
                **({'x': x_cal}
                   if getattr(stored_var_model, 'uses_x', False) else {}),
            )
        else:
            sd_at_cal = np.full_like(y_pred, median_sigma)
        sd_at_cal = np.maximum(sd_at_cal, 1e-12)
        std_residuals = residuals / sd_at_cal

        # -- Plot A: Residuals vs X with prediction interval ---------------
        # -- Plot B: Standardised residuals vs X ---------------------------
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            fig_ra, ax_ra = plt.subplots(figsize=(5, 3.5))
            ax_ra.fill_between(x_grid_res, pct_2s_lo, pct_2s_hi,
                               color="steelblue", alpha=0.15,
                               label="±2σ prediction")
            ax_ra.fill_between(x_grid_res, pct_1s_lo, pct_1s_hi,
                               color="steelblue", alpha=0.30,
                               label="±1σ prediction")
            ax_ra.axhline(0, color="grey", ls="-", lw=0.8)
            ax_ra.scatter(x_cal, residuals, c="steelblue", s=40,
                          edgecolors="k", linewidths=0.5, zorder=5)
            ax_ra.set_xlabel("X")
            ax_ra.set_ylabel("Residual  (Y − Ŷ)")
            ax_ra.set_title("Residuals vs X  (with prediction interval)")
            ax_ra.legend(fontsize=7, loc="best")
            fig_ra.tight_layout()
            st.pyplot(fig_ra)
            plt.close(fig_ra)

        with col_res2:
            fig_rb, ax_rb = plt.subplots(figsize=(5, 3.5))
            ax_rb.scatter(x_cal, std_residuals, c="steelblue", s=40,
                          edgecolors="k", linewidths=0.5, zorder=5)
            ax_rb.axhline(0, color="grey", ls="-", lw=0.8)
            ax_rb.axhline(-2, color="red", ls="--", lw=0.7, alpha=0.6)
            ax_rb.axhline(2, color="red", ls="--", lw=0.7, alpha=0.6)
            ax_rb.set_xlabel("X")
            ax_rb.set_ylabel("Standardised residual")
            ax_rb.set_title("Standardised residuals vs X")
            fig_rb.tight_layout()
            st.pyplot(fig_rb)
            plt.close(fig_rb)

        # -- Breusch–Pagan test on the STANDARDISED residuals --------------
        #    (tests whether the variance model has successfully removed
        #     heteroscedasticity)
        # -- Breusch–Pagan test on the STANDARDISED residuals --------------
        #    (tests whether the variance model has successfully removed
        #     heteroscedasticity)
        from statsmodels.stats.diagnostic import het_breuschpagan
        import statsmodels.api as sm

        exog_bp = sm.add_constant(x_cal)
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(
            std_residuals, exog_bp)

        st.markdown("##### Breusch–Pagan test on standardised residuals")
        st.markdown(
            "This tests whether the **standardised** residuals "
            "(after dividing by the model-predicted noise) still show "
            "systematic changes in spread across X. If the variance "
            "model is adequate, the standardised residuals should have "
            "roughly constant variance."
        )
        st.markdown(
            f"- LM statistic: **{bp_lm:.4f}**\n"
            f"- *p*-value: **{bp_lm_p:.4f}**"
        )
        if bp_lm_p < 0.05:
            st.warning(
                "Significant heteroscedasticity remains in the "
                "standardised residuals (*p* < 0.05). The current "
                "variance model may not fully capture how the noise "
                "changes with X. Consider a different variance equation, "
                "or try a variance-stabilising transform (e.g. log Y)."
            )
        else:
            st.success(
                "No significant heteroscedasticity in the "
                "standardised residuals (*p* ≥ 0.05). The variance "
                "model appears adequate."
            )

        # -- Guided diagnostic questions -----------------------------------
        st.markdown("---")
        st.markdown("#### Interpreting the residual plots")
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

        st.info(
            "If the answer to both of the questions above is 'no', "
            "your model appears to be adequate and you can proceed "
            "to generating inverse predictions."
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

            # -- Inverse solve method --------------------------------------
            if eq_model_r.has_symbolic_inverse:
                st.success("Symbolic (closed-form) inverse found:")
                st.latex(eq_model_r.inverse_latex_str())
            else:
                st.info("No closed-form inverse — numerical root-finding was used.")

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
                        mu_proxy, sigma_draws, var_param_draws,
                        prescribed_params=prescribed_values,
                        **({'x': np.full(n_draws, float(np.median(x_cal)))}
                           if getattr(stored_var_model, 'uses_x', False) else {}),
                    )
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

            # -- Calibration fit with inverse-prediction overlay -----------
            st.subheader("Calibration fit with inverse prediction intervals")
            from matplotlib.lines import Line2D as _Line2D
            fig_overlay, ax_ov = plt.subplots(figsize=(9, 5))
            ax_ov.scatter(x_cal, y_cal, c="black", zorder=5, s=40,
                          label="Calibration data")
            for i in np.random.choice(n_total, min(200, n_total), replace=False):
                p_i = {k: posterior[k][i] for k in eq_model_r.param_names}
                ax_ov.plot(x_grid, eq_model_r.forward_numpy(p_i, x_grid),
                           alpha=0.03, color="steelblue")
            for row in inv_rows:
                y_val_ov = row["Y_observed"]
                x_lo_ov = row[f"X_lo ({credible_level:.0%})"]
                x_hi_ov = row[f"X_hi ({credible_level:.0%})"]
                x_med_ov = row["X_median"]
                if np.isfinite(x_lo_ov) and np.isfinite(x_hi_ov):
                    ax_ov.plot([x_lo_ov, x_hi_ov], [y_val_ov, y_val_ov],
                               color="red", lw=2)
                    ax_ov.plot(x_med_ov, y_val_ov, "o", color="red", zorder=6)
                    if len(inv_rows) <= 6:
                        ax_ov.annotate(
                            f"Y={y_val_ov:.3g}",
                            xy=(x_med_ov, y_val_ov),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color="red",
                        )
            ax_ov.set_xlabel("X")
            ax_ov.set_ylabel("Y")
            ax_ov.set_title(
                f"Calibration fit with {credible_level:.0%} credible "
                f"intervals for X"
            )
            ax_ov.legend(handles=[
                _Line2D([],[], marker='o', color='black', ls='none',
                        markersize=6, label='Calibration data'),
                _Line2D([],[], color='steelblue', alpha=0.5,
                        label='Posterior curves'),
                _Line2D([],[], color='red', lw=2,
                        label=f'{credible_level:.0%} CI for X'),
                _Line2D([],[], marker='o', color='red', ls='none',
                        markersize=6, label='X median'),
            ])
            fig_overlay.tight_layout()
            st.pyplot(fig_overlay)
            plt.close(fig_overlay)

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

            # -- Download all results as ZIP --------------------------------
            import io, zipfile
            st.markdown("---")
            def _build_zip():
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    if cal_df is not None:
                        zf.writestr("calibration_data.csv",
                                    cal_df.to_csv(index=False))
                    zf.writestr("mcmc_summary.csv", summary_df.to_csv())
                    zf.writestr("inverse_predictions.csv",
                                df_result.to_csv(index=False))
                return buf.getvalue()

            st.download_button(
                "Download all calibration data and results (ZIP)",
                _build_zip(),
                file_name="calibration_results.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

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
