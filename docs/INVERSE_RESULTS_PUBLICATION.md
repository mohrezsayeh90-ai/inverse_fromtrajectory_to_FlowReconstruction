# Particle-count inverse reconstruction results

This branch adds the scientific code, raw observations, fits, diagnostics, figures and reports from Main_Proposal commit `370872d830f3b469570fcb3213cfebf6d945236a` to the FLOW repository. Its model and original research files are based on `f6b8690059716e809c2dac27c8715e65fcb1d267` (`results/soap-cosine-200k`). Existing FLOW files and earlier results are preserved.

## Read the results

- [Inverse rewrite report in Persian](INVERSE_REWRITE_FA.md)
- [Nine profile comparisons](../New_Adaptive_Particle_Results/inverse_rewrite/figures/all_profiles.pdf)
- [Nine-cycle local inlet supplement](../New_Adaptive_Particle_Results/inverse_rewrite/figures/nine30_local_inlet.pdf)
- [Recorded validation audit](../New_Adaptive_Particle_Results/inverse_rewrite/delivery_validation.json)
- [Earlier SOAP study](SOAP_ADAPTIVE_PARTICLE_COUNT_FA.md)
- [Original adaptive study](ADAPTIVE_PARTICLE_COUNT_FA.md)

The nine main comparisons use the frozen SOAP DeepONet inside the inverse optimization loop. FEM supplies synthetic reference observations and post-fit error evaluation. The local inlet supplement fits the integral particle equation directly with a nonnegative spline; DeepONet is used afterward for its full-channel replay. Its small inlet errors must not be presented as DeepONet inverse errors.

## Scientific entry points in this repository

The delivered environment used Python 3.12.14, JAX/JAXlib 0.12.0.1, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8. Rendering also uses PyMuPDF 1.26.6. Checkpoint identity is verified by the fitting scripts. From the repository root:

```sh
python -B experiments/inverse_rewrite/run.py --help
python -B experiments/inverse_rewrite/run.py --flow . beta5 skew_b3 flat30 twin_b3_r30 shoulder mixed waves3_r30 lobes4_r30 nine30
python -B experiments/inverse_rewrite/run.py --flow . --seed 53129 beta5 skew_b3 flat30 twin_b3_r30 shoulder mixed waves3_r30 lobes4_r30 nine30 --designs random hybrid
python -B experiments/inverse_rewrite/run_local.py --flow . --extend --fine-grid nine30
python -B experiments/inverse_rewrite/run_local.py --flow . --seed 53129 --extend --fine-grid nine30
```

Completed fits are reused when their reports exist. Keep the delivered results intact; use a separate checkout or an explicitly separate output location for modified experiments.

The copied experiment README retains the original Main_Proposal reproduction commands and thesis integration steps for provenance. Thesis-specific `report.py --thesis` and LaTeX commands require that repository. The final thesis source and PDF remain in [Main_Proposal](https://github.com/mohrezsayeh90-ai/Main_Proposal/tree/results/deeponet-inverse-rewrite).
