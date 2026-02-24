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
from typing import Dict, Optional
from equation_engine import EquationModel
from app_config import MODE

ALLOW_UPLOAD = (MODE == "local")

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
st.title("\U0001f52c Bayesian Calibration & Inverse Prediction")

_desc_path = pathlib.Path(__file__).parent / "description.md"
if _desc_path.exists():
    st.markdown(_desc_path.read_text())
else:
    st.markdown(
        "Define **any** calibration equation, fit it to data with Bayesian MCMC, "
        "then estimate **X** from new **Y** values with full uncertainty."
    )

# -- "Under the Hood" PDF link ---------------------------------------------
_pdf_path = pathlib.Path(__file__).parent / "docs" / "derivation.pdf"
if _pdf_path.exists():
    with open(_pdf_path, "rb") as _pdf_file:
        _pdf_bytes = _pdf_file.read()
    import base64 as _b64
    _pdf_b64 = _b64.b64encode(_pdf_bytes).decode()
    st.markdown(
        f'📄 <a href="data:application/pdf;base64,{_pdf_b64}" '
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
# Sidebar — MCMC settings
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("\u2699\ufe0f MCMC Settings")
    chains = st.number_input("Chains", 1, 8, 4)
    iter_sampling = st.number_input(
        "Iterations (sampling)", 500, 10000, 2000, step=500
    )
    iter_warmup = st.number_input(
        "Iterations (warm-up / tune)", 200, 5000, 1000, step=200
    )
    seed = st.number_input("Random seed", value=42)
    credible_level = st.slider(
        "Credible interval", min_value=0.50, max_value=0.99, value=0.95,
        step=0.01,
    )
    st.divider()
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
```
    """)
    if MODE == "local":
        st.divider()
        st.success("\U0001f5a5\ufe0f Running in **local** mode -- CSV upload enabled.")
    else:
        st.divider()
        st.info("\u2601\ufe0f Running in **cloud** mode -- manual data entry only.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Equation Editor (text input + live LaTeX preview)
# ══════════════════════════════════════════════════════════════════════════════
st.header("\u2460 Define Your Calibration Equation")
st.markdown(
    "Type your equation below using Python-style syntax. "
    "Any symbol other than `x` is treated as a parameter to estimate. "
    "See the sidebar for syntax help and examples."
)

equation_input = st.text_input(
    "\u270f\ufe0f **Equation**",
    value="y = a + b*x",
    placeholder="e.g.  y = a + b*x   or   y = a * exp(b*x) + c",
    help="Type any equation of the form  y = f(x).  "
         "Letters other than x become parameters to estimate.",
)

# --- Live parse & LaTeX preview -------------------------------------------
eq_model: Optional[EquationModel] = None

if equation_input.strip():
    try:
        eq_model = EquationModel(equation_input)

        st.latex(eq_model.latex_str())
        param_str = ", ".join(
            [f"**{p}**" for p in eq_model.param_names] + ["**sigma** (noise)"]
        )
        st.markdown(f"Parameters to estimate: {param_str}")
        if eq_model.has_symbolic_inverse:
            st.markdown("\u2705 **Symbolic inverse found:**")
            st.latex(eq_model.inverse_latex_str())
        else:
            st.markdown(
                "\u26a0\ufe0f No closed-form inverse -- will use **numerical root-finding** "
                "(works fine, just slower for large datasets)."
            )

    except ValueError as exc:
        st.error(f"\u274c {exc}")
    except Exception as exc:
        st.error(f"\u274c Unexpected error parsing equation: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Calibration Data
# ══════════════════════════════════════════════════════════════════════════════
st.header("\u2461 Calibration Data")
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
        x_text = st.text_area(
            "X values",
            value="1\n2\n3\n4\n5\n6\n7\n8\n9\n10",
            height=200,
        )
    with col2:
        y_text = st.text_area(
            "Y values",
            value="5.3\n9.1\n12.4\n16.0\n19.2\n23.1\n25.8\n29.5\n33.0\n37.2",
            height=200,
        )
    try:
        x_vals = [float(v) for v in x_text.strip().split("\n") if v.strip()]
        y_vals = [float(v) for v in y_text.strip().split("\n") if v.strip()]
        if len(x_vals) == len(y_vals) and len(x_vals) >= 2:
            cal_df = pd.DataFrame({"X": x_vals, "Y": y_vals})
        else:
            st.warning("X and Y must have the same number of values (>= 2).")
    except ValueError:
        st.error("Could not parse values -- make sure each line is a number.")

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
st.header("\u2462 New Y Values for Inverse Prediction")
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
        value="10.0\n20.0\n30.0",
        height=150,
        key="new_y_text",
    )
    try:
        y_new_vals = np.array(
            [float(v) for v in y_new_text.strip().split("\n") if v.strip()]
        )
    except ValueError:
        st.error("Could not parse -- enter one number per line.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run
# ══════════════════════════════════════════════════════════════════════════════
st.header("\u2463 Run Calibration & Inverse Prediction")

run_ready = (
    eq_model is not None
    and cal_df is not None
    and y_new_vals is not None
    and len(y_new_vals) > 0
)

if not run_ready:
    st.info(
        "Complete steps 1-3 above (valid equation, calibration data, "
        "and new Y values), then click **Run**."
    )
else:
    if st.button("Run", type="primary", use_container_width=True):

        import pymc as pm
        import arviz as az

        x_cal = cal_df["X"].values.astype(float)
        y_cal = cal_df["Y"].values.astype(float)

        # -- Build PyMC model (homoscedastic first) --------------------------
        with st.spinner("Building model..."):
            try:
                model = eq_model.build_pymc_model(x_cal, y_cal)
            except Exception as exc:
                st.error(f"\u274c Model build failed: {exc}")
                st.stop()

        # -- Sample posterior -------------------------------------------------
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

        # -- Extract posterior draws ------------------------------------------
        posterior: Dict[str, np.ndarray] = {}
        for par in eq_model.param_names:
            posterior[par] = trace.posterior[par].values.flatten()
        posterior["sigma"] = trace.posterior["sigma"].values.flatten()

        # -- Store results in session state for potential recalculation -------
        st.session_state["trace"] = trace
        st.session_state["posterior"] = posterior
        st.session_state["eq_model"] = eq_model
        st.session_state["x_cal"] = x_cal
        st.session_state["y_cal"] = y_cal
        st.session_state["y_new_vals"] = y_new_vals
        st.session_state["run_complete"] = True

    # ======================================================================
    # Display results (from session state so they survive widget changes)
    # ======================================================================
    if st.session_state.get("run_complete", False):

        import pymc as pm
        import arviz as az

        trace = st.session_state["trace"]
        posterior = st.session_state["posterior"]
        eq_model_r = st.session_state["eq_model"]
        x_cal = st.session_state["x_cal"]
        y_cal = st.session_state["y_cal"]
        y_new_vals_r = st.session_state.get("y_new_vals", y_new_vals)

        # -- MCMC Summary -----------------------------------------------------
        st.subheader("MCMC Summary")
        summary_vars = eq_model_r.param_names + ["sigma"]
        summary_df = az.summary(trace, var_names=summary_vars)
        st.dataframe(summary_df, use_container_width=True)

        # -- Calibration fit plot ----------------------------------------------
        st.subheader("Calibration Fit")
        fig_fit, ax_fit = plt.subplots(figsize=(9, 5))
        ax_fit.scatter(
            x_cal, y_cal, c="black", zorder=5, s=40, label="Data"
        )

        x_grid = np.linspace(
            x_cal.min() * 0.9, x_cal.max() * 1.1, 300
        )
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

        # ==================================================================
        # STEP 4b — Residual Diagnostics
        # ==================================================================
        st.subheader("Residual Diagnostics")

        # Compute residuals using posterior median parameters
        median_params = {p: np.median(posterior[p]) for p in eq_model_r.param_names}
        y_pred = eq_model_r.forward_numpy(median_params, x_cal)
        residuals = y_cal - y_pred

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            # Residuals vs fitted
            fig_rvf, ax_rvf = plt.subplots(figsize=(5, 3.5))
            ax_rvf.scatter(y_pred, residuals, c="steelblue", s=40, edgecolors="k", linewidths=0.5)
            ax_rvf.axhline(0, color="red", ls="--", lw=1)
            ax_rvf.set_xlabel("Fitted values (Ŷ)")
            ax_rvf.set_ylabel("Residuals (Y − Ŷ)")
            ax_rvf.set_title("Residuals vs Fitted")
            fig_rvf.tight_layout()
            st.pyplot(fig_rvf)
            plt.close(fig_rvf)

        with col_res2:
            # Residuals vs X
            fig_rvx, ax_rvx = plt.subplots(figsize=(5, 3.5))
            ax_rvx.scatter(x_cal, residuals, c="steelblue", s=40, edgecolors="k", linewidths=0.5)
            ax_rvx.axhline(0, color="red", ls="--", lw=1)
            ax_rvx.set_xlabel("X")
            ax_rvx.set_ylabel("Residuals (Y − Ŷ)")
            ax_rvx.set_title("Residuals vs X")
            fig_rvx.tight_layout()
            st.pyplot(fig_rvx)
            plt.close(fig_rvx)

        # -- Statistical tests for residual randomness -----------------------
        st.markdown("**Statistical tests for residual structure:**")

        from statsmodels.stats.diagnostic import het_breuschpagan
        from scipy.stats import norm as _norm

        # Breusch-Pagan test for heteroscedasticity
        import statsmodels.api as sm
        exog_bp = sm.add_constant(y_pred)
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, exog_bp)

        # Wald-Wolfowitz runs test for randomness (manual implementation)
        def _wald_wolfowitz_runs_test(residuals_arr):
            """Two-sided Wald-Wolfowitz runs test on the signs of residuals."""
            signs = np.array(residuals_arr) > 0
            n_pos = int(signs.sum())
            n_neg = int((~signs).sum())
            n = n_pos + n_neg
            if n_pos == 0 or n_neg == 0 or n < 3:
                return np.nan, np.nan
            runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
            e_runs = 1.0 + (2.0 * n_pos * n_neg) / n
            var_runs = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n**2 * (n - 1.0))
            if var_runs <= 0:
                return np.nan, np.nan
            z = (runs - e_runs) / np.sqrt(var_runs)
            p_value = 2.0 * _norm.sf(np.abs(z))
            return z, p_value

        runs_stat, runs_p = _wald_wolfowitz_runs_test(residuals)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Breusch-Pagan test (heteroscedasticity)")
            st.markdown(
                f"- LM statistic: **{bp_lm:.4f}**\n"
                f"- *p*-value: **{bp_lm_p:.4f}**"
            )
            if bp_lm_p < 0.05:
                st.warning(
                    "⚠️ Significant heteroscedasticity detected (*p* < 0.05). "
                    "The variance of the residuals is not constant across fitted values."
                )
            else:
                st.success("✅ No significant heteroscedasticity (*p* ≥ 0.05).")

        with col_t2:
            st.markdown("##### Wald–Wolfowitz runs test (randomness)")
            st.markdown(
                f"- Test statistic: **{runs_stat:.4f}**\n"
                f"- *p*-value: **{runs_p:.4f}**"
            )
            if runs_p < 0.05:
                st.warning(
                    "⚠️ Significant non-randomness in residuals (*p* < 0.05). "
                    "There may be systematic structure the model does not capture."
                )
            else:
                st.success("✅ Residuals appear random (*p* ≥ 0.05).")

        # -- Guided diagnostic questions & practical pointers ----------------
        st.markdown("---")
        st.markdown("#### 🔍 Interpreting the residual plots")
        st.markdown(
            "Use the plots and test results above to check whether your "
            "model is a good fit. Here are some questions to guide you:"
        )

        with st.expander("**Can you see a pattern (curve, trend) in the residuals?**", expanded=False):
            st.markdown(
                "If the residuals show a systematic curve or trend rather than "
                "a random scatter around zero, the current equation may not capture "
                "the true relationship between X and Y.\n\n"
                "**What to try:**\n"
                "- Go back to **Step ①** and try a different model — for example, "
                "add a quadratic term (`y = a + b*x + c*x**2`), or switch to a "
                "nonlinear form like `y = a * exp(b*x)` or `y = a * x**b`.\n"
                "- If you already used a polynomial, try increasing the order by one.\n"
                "- Look at the scatter plot of your calibration data for visual clues "
                "about what shape the curve should be."
            )

        with st.expander("**Does the spread of residuals change systematically (bigger at one end)?**", expanded=False):
            st.markdown(
                "If the residuals fan out — e.g. small residuals at low X (or Ŷ) "
                "and large residuals at high X — the noise is not constant "
                "(heteroscedasticity). The Breusch-Pagan test above checks for "
                "this formally.\n\n"
                "**What to try:**\n"
                "- **Log-transform your data** before fitting. Replace your Y values "
                "with `log(Y)` and fit the model to the transformed data. This often "
                "stabilises the variance when noise is proportional to the signal.\n"
                "- Alternatively, try a **power transform** such as `sqrt(Y)` or "
                "fit a model in log-space, e.g. `y = a + b*log(x)`.\n"
                "- If your assay has a known coefficient-of-variation (CV), working "
                "in log-space naturally accounts for constant-CV noise."
            )

        with st.expander("**Are there one or two outliers far from zero?**", expanded=False):
            st.markdown(
                "Isolated large residuals may indicate data entry errors, sample "
                "preparation problems, or genuine outliers.\n\n"
                "**What to try:**\n"
                "- Double-check the raw data for transcription errors.\n"
                "- If the outlier is real, consider whether that standard should be "
                "excluded or whether a more robust model is needed.\n"
                "- As a quick sanity check, remove the suspect point from your "
                "calibration data and re-run — if the fit improves dramatically, "
                "that point was likely problematic."
            )

        with st.expander("**Do the residuals look random and centred around zero?**", expanded=False):
            st.markdown(
                "Great — this is what you want to see! If both statistical tests "
                "also pass (*p* ≥ 0.05), the model assumptions appear to be "
                "satisfied and you can be confident in the inverse predictions below."
            )

        # ==================================================================
        # STEP 5 — Inverse Prediction
        # ==================================================================
        st.subheader("Inverse Predictions")
        alpha_tail = (1 - credible_level) / 2
        sigma_draws = posterior["sigma"]
        n_draws = len(sigma_draws)
        x_hint = float(x_cal.mean())
        x_lo_range = float(x_cal.min()) - 3 * float(np.ptp(x_cal))
        x_hi_range = float(x_cal.max()) + 3 * float(np.ptp(x_cal))

        inv_rows = []
        all_draws = []

        progress_bar = st.progress(0, text="Computing inverse predictions...")
        for yi, y_val in enumerate(y_new_vals_r):
            y_star = y_val + np.random.randn(n_draws) * sigma_draws
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
                text=f"Inverse prediction {yi + 1}/{len(y_new_vals_r)}"
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

        # -- Posterior histograms ----------------------------------------------
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
                ax.hist(
                    x_draws, bins=60, density=True,
                    color="steelblue", alpha=0.7,
                )
                lo = np.percentile(x_draws, 100 * alpha_tail)
                hi = np.percentile(x_draws, 100 * (1 - alpha_tail))
                ax.axvline(
                    lo, color="red", ls="--",
                    label=f"{credible_level:.0%} CI",
                )
                ax.axvline(hi, color="red", ls="--")
                ax.axvline(
                    np.median(x_draws), color="orange", label="Median"
                )
                ax.legend(fontsize=8)
            ax.set_title(f"Y = {y_val:.4g}")
            ax.set_xlabel("X")

        for idx_y in range(n_new, n_rows_fig * n_cols):
            axes_inv[idx_y // n_cols][idx_y % n_cols].set_visible(False)

        fig_inv.suptitle(
            "Inverse-prediction posterior distributions", fontsize=14
        )
        fig_inv.tight_layout()
        st.pyplot(fig_inv)
        plt.close(fig_inv)

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<center><small>Bayesian Calibration Tool -- Powered by "
    "<a href='https://www.pymc.io/'>PyMC</a>, "
    "<a href='https://www.sympy.org/'>SymPy</a> & "
    "<a href='https://streamlit.io/'>Streamlit</a></small></center>",
    unsafe_allow_html=True,
)
