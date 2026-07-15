# Folder marker for RFT.typo.v7 package.
#
# v7 reuses v5's task/prompts/response_format (RFT.typo.v5._task) and
# v6's eval split, but introduces:
#   * resampled train (positive columns upsampled 2x) to break the
#     abstain-attractor that v6 fell into;
#   * three reward variants (r1=additive, r2=harmonic, r3=weighted)
#     defined in ``RFT.typo.v7._grader_variants``.
#
# Hyperparameters fixed to the v6 best recipe: n_epochs=1, lr=1.0,
# compute=1.0 (the "baseline" config from v6).
