# Experiment Design

1. Generate ground-truth 1D signals (sum of sinusoids, multisine, piecewise).
2. Build ill-conditioned forward operators (Gaussian blur, downsampling, rank-deficient matrix).
3. Add Gaussian noise at low/medium/high levels to form measurements.
4. Reconstruct with pseudoinverse, Tikhonov, and TSVD; sweep hyperparameters.
5. Diagnose with singular value decay, condition numbers, Picard plots, and L-curves.
6. Select reasonable regularization settings (LLM-guided step deferred).
