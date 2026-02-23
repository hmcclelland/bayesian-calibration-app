Welcome to the **Bayesian Calibration & Inverse Prediction Tool**.

Assays are usually calibrated by fitting instrument signal (Y—e.g., fluorescence) against known standards (X—e.g., metabolite concentration). That works for predicting **Y from X**, because the error model is defined in that direction. But in practice we do the reverse: **we measure a new Y and want to estimate X**. Standard frequentist regression doesn’t naturally give uncertainty for X, so we need an **inverse prediction interval**—the range of X values that could reasonably have produced the observed Y. This app makes it straightforward: pick a model that matches your assay, fit it to your calibration dataset, then enter new Y values to get X estimates with Bayesian credible intervals. Use the web version for manual entry, or run it locally and upload a CSV.

Developed by the McClelland Lab, UCL.
