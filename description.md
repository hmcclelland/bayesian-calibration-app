**Welcome to CaliBR — the Calibration with Bayesian inverse Regression tool.**

Assays and proxies are calibrated by regressing instrument signals (Y—e.g., fluorescence, geochemical compositions) against known standards or conditions (X—e.g., metabolite concentrations, temperatures). When using calibrations to infer X from Y, we need to ask: "What values of X could reasonably have given rise to this new observed **Y**?". Conventional frequentist regression isn’t built for this, but the question is naturally answered with a Bayesian approach.

**CaliBR** (_“calibre”_) makes this easy. Choose a model appropriate for your assay, fit it to your calibration dataset, and then input a new Y to obtain a posterior estimate of X with a credible interval (or an inverse prediction interval; IPI). Use the web interface for manual entry, or download the tool to import calibration data from a `.csv` file.

_Developed by the McClelland Lab, UCL._