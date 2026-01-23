# Bayesian calibration of a biology-based mathematical model using serial tumor volume and [$^{89}$Zr]-CD8 PET to predict the response of triple-negative breast cancer

**Authors:** Guilherme Rodrigues, Paulo F. A. Mancera, Anna G. Sorace, Patrick N. Song, Thomas E. Yankeelov, Ernesto A. B. F. Lima

This repository implements a Bayesian framework for calibrating, selecting, and validating mechanistic ODE-based models of tumor growth and treatment response. The pipeline supports MCMC-based parameter inference with uncertainty quantification, compares alternative model formulations using parsimony-driven criteria (e.g., BIC/BIC weights), and evaluates predictive performance via global testing and leave-one-out validation. It is designed to capture inter-individual heterogeneity in triple-negative breast cancer, including radiation-sensitive and radiation-resistant phenotypes, and can incorporate functional imaging biomarkers (immuno-PET SUV) to inform treatment-effect terms.
