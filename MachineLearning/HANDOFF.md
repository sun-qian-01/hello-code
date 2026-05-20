# Cardinality Estimation Handoff

## Goal

This folder contains a long-running attempt to improve leaderboard score for the SQL cardinality estimation task.

The working objective is:

- Input: simplified SQL query fields from `train.csv` / `test.csv`
- Output: predicted `Cardinality`
- Metric: Mean Q-Error

The user reported an online score of about `4.88061` on the original submission, then iterated through many experimental routes. Recent online scores confirmed the value-target-encoding direction: `submission_value_te_id_heavy_on_isotonic.csv` scored `3.24732`, the more aggressive `submission_value_te_column_capped_on_isotonic.csv` scored `3.15977`, and `submission_value_te_capped_pair02_on_isotonic.csv` scored `3.15321`. Pair-token follow-ups were marginal: `submission_value_te_midopt_pair04_on_isotonic.csv` scored `3.15393`, and `submission_value_te_opt6_pair04_on_isotonic.csv` scored `3.16529`. The best local OOF result reached so far is about `3.47514`, but online gains have often been smaller than local improvements, so overfitting to local OOF is still a risk.

## Dataset Shape

Files:

- `train.csv`: 50,000 rows
- `test.csv`: 5,000 rows
- `column_min_max_vals.csv`: min/max/cardinality/unique stats
- `schematext.sql`: schema description

Schema pattern:

- `title t` is the central table
- `movie_companies mc`, `cast_info ci`, `movie_info mi`, `movie_info_idx mi_idx`, `movie_keyword mk` all join to `title` via `movie_id`

Important empirical facts discovered during exploration:

- Test templates are highly repetitive relative to train.
- Exact predicate shape coverage is high:
  about 4,792 of 5,000 test rows have an exact training shape match.
- Column-set coverage is almost complete:
  only about 1 test row is unseen at the `table + predicate columns` level.
- That means local template models are viable.

## Best Current Submission Candidates

These are the most relevant files to try first.

1. `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`
   Conservative exact-shape residual surface gate:
   local OOF about `3.47604`; changes 470 test rows versus
   `submission_value_te_capped_pair02_on_isotonic.csv`.

2. `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`
   More aggressive exact-shape residual surface gate:
   local OOF about `3.47514`; changes 1402 test rows versus
   `submission_value_te_capped_pair02_on_isotonic.csv`.

3. `submission_shape_surface_m50_d2_a100_s0p3_g0p0.csv`
   Slightly more regularized shape surface:
   local OOF about `3.47599`; changes 1353 test rows versus
   `submission_value_te_capped_pair02_on_isotonic.csv`.

4. `submission_value_te_capped_pair02_on_isotonic.csv`
   Current online-known best:
   online `3.15321`, local OOF about `3.48296`.

5. `submission_value_te_midopt_pair04_on_isotonic.csv`
   Pair-token follow-up that was nearly flat online:
   online `3.15393`, local OOF about `3.48058`.

6. `submission_value_te_column_capped_on_isotonic.csv`
   Previous online-known best before pair02:
   online `3.15977`, local OOF about `3.48660`.

7. `submission_value_te_id_heavy_on_isotonic.csv`
   Earlier conservative value-target-encoding candidate:
   online `3.24732`, local OOF about `3.54801`.

8. `submission_value_te_column_opt6_on_isotonic.csv`
   Column-weight-only local optimization:
   local OOF about `3.48539`; changes 1278 test rows versus current online best.

9. `submission_isotonic_family.csv`
   Previous online-known best before value target encoding:
   online `3.38517`, local OOF about `3.67115`.

10. `submission_isotonic_stack_gate.csv`
   Table-level conservative gate over stacked and isotonic predictions:
   about `3.66998`

11. `submission_stacked_family_gate.csv`
   Conservative stacked family gate:
   about `3.67412`

12. `submission_hist_family_gate.csv`
   HistGradientBoosting residual expert routed by family:
   about `3.67469`

10. `submission_meta_gate_family_fallback.csv`
   Previous best local OOF:
   about `3.67583`

11. `submission_meta_gate_huber_resid.csv`
   Tiny residual meta-model on top of the gate:
   about `3.67916`

12. `submission_column_family_gate_shape50_cols80_affine.csv`
   Conservative affine-calibrated gate:
   about `3.67961`

13. `submission_column_family_gate_shape50_cols80.csv`
   Best uncalibrated nested-gate OOF:
   about `3.67994`

14. `submission_column_family_gate_shape_cols.csv`
   Slightly more conservative two-level gate:
   about `3.68027`

15. `submission_column_family_gate.csv`
   Column-family gate with wider expert pool:
   about `3.68062`

16. `submission_column_family_gate_core.csv`
   Conservative column-family gate using only strongest late-stage experts:
   about `3.68616`

17. `submission_latent_final_blend.csv`
   Previous best local OOF:
   about `3.69370`

18. `submission_lowrank_latent.csv`
   Low-rank latent residual model only:
   about `3.69431`

19. `submission_rewrite_angle_affine.csv`
   Best pre-latent template-first route:
   about `3.70057`

20. `submission_grid_surface.csv`
   Template surface reconstruction route:
   about `3.70169`

All of these files were checked for:

- exactly 5,000 rows
- columns `Id,Cardinality`
- unique IDs
- positive integer predictions

## Recommended Starting Point

If continuing immediately, start from:

- `submission_isotonic_family.csv`
- `isotonic_family_model.py`
- `submission_value_te_id_heavy_on_isotonic.csv`
- `submission_value_te_column_core_on_isotonic.csv`
- `value_residual_on_isotonic_family.py`
- `value_residual_target_encoding.py`
- `isotonic_stack_gate.py`
- `submission_stacked_family_gate.csv`
- `stacked_family_gate.py`
- `hist_residual_family_gate.py`
- `submission_meta_gate_family_fallback.csv`
- `meta_gate_family_fallback.py`
- `meta_gate_residual.py`
- `column_family_gate.py`
- `lowrank_latent_residual.py`
- `hierarchical_template_model.py`
- `HANDOFF.md`

That combination represents the strongest current route plus the best previous fallback.
Because online gains have lagged local OOF gains before, also keep
`submission_stacked_family_gate.csv`,
`submission_hist_family_gate.csv`,
`submission_meta_gate_huber_resid.csv`,
`submission_column_family_gate_shape50_cols80_affine.csv`,
`submission_column_family_gate_core.csv`, and `submission_latent_final_blend.csv`
as safer comparison submissions.

## Experiment Timeline

### 1. Baseline LightGBM parsing route

Files:

- `cardinality_estimation.py`
- `fast_model.py`
- `improved_model.py`
- `tune_model.py`

Idea:

- parse tables / joins / predicates
- create per-column predicate features
- train global LightGBM on `log1p(Cardinality)`

Outcome:

- decent local validation
- insufficient online score
- too dependent on global regression assumptions

### 2. Strong global ensemble route

Files:

- `strong_model.py`
- `final_cardinality_model.py`
- `final_oof_predictions.csv`
- `submission.csv`
- `submission_no_smallblend.csv`

Idea:

- better engineered SQL features
- multiple LightGBM regressors
- small-cardinality classifiers
- OOF-based Q-Error calibration

Outcome:

- improved online score materially
- still plateaued far behind stronger leaderboard entries
- good fallback baseline

### 3. Group residual / local expert route

Files:

- `group_expert_oof.csv`
- `submission_group_expert.csv`
- `residual_blend_oof.csv`
- `submission_residual_blend.csv`
- `submission_superblend_affine.csv`
- `submission_superblend_convex.csv`

Idea:

- use current best prediction as baseline
- train residual models by table combo / predicate shape
- blend them using OOF Q-Error

Outcome:

- helpful but small gains
- online transfer was weaker than local OOF suggested

### 4. Template-first rewrite

Files:

- `hierarchical_template_model.py`
- `hierarchical_template_oof.csv`
- `submission_hierarchical_template.csv`
- `submission_rewrite_angle_affine.csv`
- `submission_rewrite_angle_convex.csv`
- `submission_rewrite_hier_only_soft.csv`
- `submission_rewrite_hier_group.csv`
- `submission_rewrite_hier_superlike.csv`

Idea:

- stop treating the problem as one global regressor
- split by template hierarchy:
  `exact shape -> cols -> tableop -> table -> fallback`
- fit local models per template

Outcome:

- first genuinely different modeling angle
- local OOF improved to roughly `3.70057`
- still only modest online gain

### 5. Monotone and physical estimator routes

Files:

- `monotone_template_model.py`
- `submission_monotone_template.csv`
- `join_distribution_estimator.py`
- `submission_join_distribution_blend.csv`
- `submission_join_physical_only.csv`
- `submission_empirical_single_table.csv`

Idea:

- enforce monotonicity for `<` / `>` thresholds within fixed templates
- try star-join physical estimates using table selectivities, child coverage, fanout
- try empirical CDF for single-table predicates

Outcome:

- intellectually useful
- practically weak
- physical estimator did not carry enough signal without real joint distribution information

### 6. Template surface reconstruction

Files:

- `grid_surface_model.py`
- `grid_surface_oof.csv`
- `submission_grid_surface.csv`

Idea:

- for high-frequency exact templates, learn cardinality surface directly from value coordinates
- use KNN / ND interpolation

Outcome:

- slightly useful
- still not enough on its own

### 7. Heavy latent join-key inversion

Files:

- `latent_join_inversion.py`
- `fast_latent_join.py`

Idea:

- assume hidden movie clusters
- simulate join cardinality as sum over latent clusters

Outcome:

- too slow in this environment without torch/jax/tensorflow
- no usable submission artifact from these direct simulators

### 8. Matrixized low-rank latent residual route

Files:

- `lowrank_latent_residual.py`
- `lowrank_latent_oof.csv`
- `submission_lowrank_latent.csv`
- `latent_final_blend_oof.csv`
- `submission_latent_final_blend.csv`
- `submission_lowrank_grid.csv`
- `submission_lowrank_hier.csv`
- `submission_mostly_lowrank.csv`

Idea:

- approximate hidden join-key latent factors using hashed template / value / pairwise interaction tokens
- use `FeatureHasher` + `TruncatedSVD` to get latent dimensions
- learn residuals of best current model in that latent space

Outcome:

- best current local route
- `submission_latent_final_blend.csv` is current top local candidate

### 9. Column-family expert gate

Files:

- `column_family_gate.py`
- `column_family_gate_shape50_cols80_oof.csv`
- `submission_column_family_gate_shape50_cols80.csv`
- `column_family_gate_shape_cols_oof.csv`
- `submission_column_family_gate_shape_cols.csv`
- `column_family_gate_oof.csv`
- `submission_column_family_gate.csv`
- `column_family_gate_core_oof.csv`
- `submission_column_family_gate_core.csv`

Idea:

- existing experts are highly correlated globally but win on different
  `table combo + predicate columns` families
- use a second-level gate with an outer 5-fold split
- choose the best expert for each exact-shape / column-family group using only
  the fold's training side, then apply that choice to validation/test

Outcome:

- best local nested-gate score reached about `3.67994`
- strongest candidate is `submission_column_family_gate_shape50_cols80.csv`
- this is an expert-selection improvement rather than a new base model, so
  public/private transfer should be checked carefully

### 10. Tiny residual meta-model after gate

Files:

- `meta_gate_residual.py`
- `meta_gate_huber_resid_oof.csv`
- `submission_meta_gate_huber_resid.csv`
- `column_family_gate_shape50_cols80_affine_oof.csv`
- `submission_column_family_gate_shape50_cols80_affine.csv`

Idea:

- start from the best gate prediction
- apply a global affine calibration to the gate
- train a tiny Huber residual model using existing expert prediction logs,
  expert disagreement, table one-hot, and simple predicate-count/value features
- clip residual target to `[-1, 1]` and shrink predictions by `0.05`

Outcome:

- affine gate improved local OOF to about `3.67961`
- Huber residual meta-model improved local OOF to about `3.67916`
- the submission changes are small relative to the affine gate, but this is
  still a post-processing layer and should be online-validated cautiously

### 11. Family-level fallback after meta model

Files:

- `meta_gate_family_fallback.py`
- `meta_gate_family_fallback_oof.csv`
- `submission_meta_gate_family_fallback.csv`

Idea:

- use `submission_meta_gate_huber_resid.csv` as the default prediction
- do not add more model capacity
- for `table + predicate columns` families with at least `80` training rows,
  fall back to `submission_latent_final_blend.csv` whenever latent beats meta
  by more than a tiny margin on the training side of each fold

Outcome:

- local OOF improved to about `3.67583`
- only about `1115` of `5000` test rows changed relative to
  `submission_meta_gate_huber_resid.csv`
- this is the most efficient recent improvement because it removes unstable
  corrections rather than fitting another richer model

### 12. HistGradientBoosting residual expert and stacked family gate

Files:

- `hist_residual_family_gate.py`
- `hist_residual_oof.csv`
- `submission_hist_residual.csv`
- `hist_family_gate_oof.csv`
- `submission_hist_family_gate.csv`
- `stacked_family_gate.py`
- `stacked_family_gate_oof.csv`
- `submission_stacked_family_gate.csv`

Idea:

- switch algorithm family from LightGBM/gates to sklearn
  `HistGradientBoostingRegressor`
- train a global residual model on top of `submission_meta_gate_family_fallback`
- use the hist residual prediction as a new expert in a conservative
  `table + predicate columns` family router
- stack the best post-processing experts with a second family router, defaulting
  to `hist_family_gate` and only switching large stable families

Outcome:

- standalone hist residual did not beat mean OOF, but improved tail percentiles
- hist family gate improved local OOF to about `3.67469`
- stacked family gate improved local OOF to about `3.67412`
- `submission_stacked_family_gate.csv` changes only about `404` rows versus
  `submission_meta_gate_family_fallback.csv`, so it is a conservative candidate

### 13. Column-family isotonic residual model

Files:

- `isotonic_family_model.py`
- `isotonic_family_oof.csv`
- `submission_isotonic_family.csv`
- `isotonic_stack_gate.py`
- `isotonic_stack_gate_oof.csv`
- `submission_isotonic_stack_gate.csv`

Idea:

- stop treating the next step as pure expert routing
- within each `table + predicate columns` family, compute a 1D selectivity score:
  the sum of log per-predicate selectivities from column min/max stats
- fit an `IsotonicRegression` residual curve over that score
- only apply it to large families with at least `400` training rows, using
  shrink `0.35` on top of `submission_stacked_family_gate.csv`
- then optionally use a table-level gate that switches from stacked to isotonic
  only when a table-combination has at least `800` rows and isotonic beats
  stacked by margin `0.01` on the training side

Outcome:

- local OOF improved to about `3.67115`
- the table-level `isotonic_stack_gate` reproduces local OOF about `3.66998`
- coverage is about `44.5%` of train and `46.1%` of test rows
- this changes about `2180` test rows versus `submission_stacked_family_gate.csv`
- it is a more structural algorithmic change than another route over existing
  submissions, but should still be online-validated against the conservative
  `submission_stacked_family_gate.csv`
- the user reported `submission_isotonic_family.csv` online score `3.38517`,
  slightly better than `submission_meta_gate_huber_resid.csv` at `3.38593`

### 14. Value-level residual target encoding

Files:

- `value_residual_target_encoding.py`
- `value_residual_on_isotonic_family.py`
- `value_te_eq_conservative_on_isotonic_oof.csv`
- `submission_value_te_eq_conservative_on_isotonic.csv`
- `value_te_eq_on_isotonic_oof.csv`
- `submission_value_te_eq_on_isotonic.csv`
- `value_te_id_heavy_on_isotonic_oof.csv`
- `submission_value_te_id_heavy_on_isotonic.csv`
- `value_te_column_core_on_isotonic_oof.csv`
- `submission_value_te_column_core_on_isotonic.csv`
- `value_te_column_capped_on_isotonic_oof.csv`
- `submission_value_te_column_capped_on_isotonic.csv`
- `value_te_capped_pair02_on_isotonic_oof.csv`
- `submission_value_te_capped_pair02_on_isotonic.csv`
- `value_te_capped_pair04_on_isotonic_oof.csv`
- `submission_value_te_capped_pair04_on_isotonic.csv`
- `value_te_midopt_pair04_on_isotonic_oof.csv`
- `submission_value_te_midopt_pair04_on_isotonic.csv`
- `value_te_opt6_pair04_on_isotonic_oof.csv`
- `submission_value_te_opt6_pair04_on_isotonic.csv`

Idea:

- stop treating equality predicates as uniform selectivity
- learn cross-fitted residual maps for concrete equality values such as
  `mk.keyword_id=...`, `mc.company_id=...`, and `t.production_year=...`
- smooth each token effect toward zero with count smoothing, fit maps only on
  the training side of each fold, and apply clipped residual corrections
- produce both broad equality-value variants and column-weighted variants that
  emphasize high-cardinality entity IDs and avoid columns that hurt OOF
- after online feedback confirmed the aggressive column-weighted version, add a
  small pair-token residual (`pair_table`) on top of the capped column model

Outcome:

- online feedback:
  `submission_value_te_id_heavy_on_isotonic.csv` scored `3.24732`;
  `submission_value_te_column_capped_on_isotonic.csv` scored `3.15977`
- direct-on-isotonic variants preserve the latest online-validated base while
  adding only the new value-hotness signal
- `submission_value_te_eq_conservative_on_isotonic.csv`:
  local OOF about `3.60800`, changes 2222 test rows versus isotonic
- `submission_value_te_eq_on_isotonic.csv`:
  local OOF about `3.57583`, changes 2235 test rows versus isotonic
- `submission_value_te_id_heavy_on_isotonic.csv`:
  local OOF about `3.54801`, changes only 193 test rows versus isotonic
- `submission_value_te_column_core_on_isotonic.csv`:
  local OOF about `3.49475`, changes 1351 test rows versus isotonic
- `submission_value_te_column_capped_on_isotonic.csv`:
  local OOF about `3.48660`, changes 1351 test rows versus isotonic, online
  `3.15977`
- `submission_value_te_capped_pair02_on_isotonic.csv`:
  local OOF about `3.48296`, changes 1209 test rows versus current online best
- `submission_value_te_capped_pair04_on_isotonic.csv`:
  local OOF about `3.48115`, changes 1228 test rows versus current online best
- `submission_value_te_midopt_pair04_on_isotonic.csv`:
  local OOF about `3.48058`, changes 2015 test rows versus current online best
- `submission_value_te_opt6_pair04_on_isotonic.csv`:
  local OOF about `3.48056`, changes 2040 test rows versus current online best
- the strongest local non-`on_isotonic` variant is
  `submission_value_te_column_capped.csv`, local OOF about `3.48444`, but it
  also changes the base to `isotonic_stack_gate`; use it only as a later probe
- multi-seed checks showed the equality-value signal is stable across KFold
  seeds; however, this is a much larger local jump than previous online gains,
  so it must be leaderboard-validated carefully

### 15. Exact-shape residual surface gate

Files:

- `shape_surface_residual.py`
- `shape_surface_m50_d2_a10_s0p3_g0p0_oof.csv`
- `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`
- `shape_surface_m50_d2_a10_s0p3_g0p01_oof.csv`
- `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`
- `shape_surface_m50_d2_a100_s0p3_g0p0_oof.csv`
- `submission_shape_surface_m50_d2_a100_s0p3_g0p0.csv`

Idea:

- move away from more token target-encoding tweaks after pair tokens showed
  only marginal online improvement
- within each exact predicate shape, fit a low-dimensional continuous residual
  surface over normalized predicate values using polynomial Ridge
- gate the surface at exact-shape level using OOF Q-error; unstable shapes fall
  back to `submission_value_te_capped_pair02_on_isotonic.csv`

Outcome:

- `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`:
  local OOF about `3.47604`, only 470 test rows changed versus pair02
- `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`:
  local OOF about `3.47514`, 1402 test rows changed versus pair02
- `submission_shape_surface_m50_d2_a100_s0p3_g0p0.csv`:
  local OOF about `3.47599`, 1353 test rows changed versus pair02
- this is a more genuinely different algorithm than the pair-token variants;
  submit the conservative `g0p01` first, then the stronger `g0p0` if it helps

## Best Current Local Results

Approximate OOF means observed:

- `submission.csv` route:
  about `3.74899`

- `submission_rewrite_angle_affine.csv`:
  about `3.70057`

- `submission_lowrank_latent.csv`:
  about `3.69431`

- `submission_latent_final_blend.csv`:
  about `3.69370`

- `submission_column_family_gate_core.csv`:
  about `3.68616`

- `submission_column_family_gate.csv`:
  about `3.68062`

- `submission_column_family_gate_shape_cols.csv`:
  about `3.68027`

- `submission_column_family_gate_shape50_cols80.csv`:
  about `3.67994`

- `submission_column_family_gate_shape50_cols80_affine.csv`:
  about `3.67961`

- `submission_meta_gate_huber_resid.csv`:
  about `3.67916`

- `submission_meta_gate_family_fallback.csv`:
  about `3.67583`

- `submission_hist_family_gate.csv`:
  about `3.67469`

- `submission_stacked_family_gate.csv`:
  about `3.67412`

- `submission_isotonic_stack_gate.csv`:
  about `3.66998`

- `submission_isotonic_family.csv`:
  about `3.67115`

- `submission_value_te_eq_conservative_on_isotonic.csv`:
  about `3.60800`

- `submission_value_te_eq_on_isotonic.csv`:
  about `3.57583`

- `submission_value_te_id_heavy_on_isotonic.csv`:
  about `3.54801`

- `submission_value_te_column_core_on_isotonic.csv`:
  about `3.49475`

- `submission_value_te_column_capped_on_isotonic.csv`:
  about `3.48660`

- `submission_value_te_capped_pair02_on_isotonic.csv`:
  about `3.48296`

- `submission_value_te_capped_pair04_on_isotonic.csv`:
  about `3.48115`

- `submission_value_te_capped_pair06_on_isotonic.csv`:
  about `3.48113`

- `submission_value_te_midopt_pair04_on_isotonic.csv`:
  about `3.48058`

- `submission_value_te_opt6_pair04_on_isotonic.csv`:
  about `3.48056`

- `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`:
  about `3.47604`

- `submission_shape_surface_m50_d2_a100_s0p3_g0p0.csv`:
  about `3.47599`

- `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`:
  about `3.47514`

- `submission_value_te_column_capped.csv`:
  about `3.48444`

Important:

- online scores improved much less than local OOF
- small OOF improvements often translated into only ~0.01 online
- the target-encoding improvements are much larger locally than previous
  changes; do not assume the full OOF gain will transfer
- treat local OOF as directional, not authoritative

## What Failed or Underperformed

These routes probably should not be revisited first:

- bigger global LightGBM tuning only
- simple KNN over same-shape queries
- naive star-join physical estimator with uniform fanout assumptions
- single-table empirical CDF alone
- heavy direct latent simulator in pure Python
- template monotonic model as a main solution

## Current Interpretation

The most important conclusion from the full exploration:

- ordinary ML feature engineering has likely been squeezed close to its limit here
- the remaining gap is probably not “better booster tuning”
- the gap likely comes from recovering hidden joint structure behind `movie_id`

In practice, the strongest new signal came from:

- low-rank latent features over hashed template/value interactions

That suggests the hidden structure is real, but our current approximation is still coarse.

2026-05-19 update:

- `submission_isotonic_family.csv` transferred online in the right direction:
  online score `3.38517`, slightly better than the previous `3.38593`.
- The biggest new local signal is no longer another booster or expert router.
  It is concrete predicate-value hotness: equality predicates are not uniform,
  and repeated values such as `mk.keyword_id=...`, `mc.company_id=...`,
  `t.production_year=...`, and `mi.info_type_id=...` carry strong residual
  information.
- The new target-encoding models are cross-fitted and stable across several
  KFold seeds, but the local jump is much larger than previous online gains.
  Treat these as leaderboard candidates to validate carefully, not guaranteed
  private-score improvements.

## What To Try Next

If someone continues this work, the best next steps are:

1. Online-validate the exact-shape residual surface candidates now that pair
   tokens have plateaued. Recommended order:
   `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`,
   then `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`, then
   `submission_shape_surface_m50_d2_a100_s0p3_g0p0.csv`.

2. If the value-encoding route transfers, tune it with online feedback:
   shrink per column, cap maximum log correction, and separate public/private
   risk between high-cardinality IDs and low-cardinality fields.

3. Push the low-rank latent route further, but efficiently.
   Use smaller/faster SVD sweeps and better residual experts rather than direct Python simulators.

4. Move from global latent factors to per-family latent factors.
   Separate latent spaces for:
   - `title + cast_info`
   - `title + movie_companies`
   - `title + movie_info`
   - `title + movie_keyword`
   and the 3-table combinations built from them.

5. Use stronger interaction tokenization.
   Especially:
   - pairwise predicate-bin interactions
   - table-specific value-bin crosses
   - shape-conditioned latent features

6. If external data ever becomes available, reconstruct actual table-level distributions.
   That would likely dominate all current CSV-only approaches.

## File Map

Core scripts worth understanding:

- `final_cardinality_model.py`
  strongest classical ensemble baseline

- `hierarchical_template_model.py`
  strongest template-first rewrite before latent route

- `lowrank_latent_residual.py`
  strongest current latent-factor approximation

- `grid_surface_model.py`
  useful reference for direct template-surface fitting

- `column_family_gate.py`
  current best post-processing / expert-selection route

- `meta_gate_residual.py`
  current best tiny residual model on top of the gate

- `meta_gate_family_fallback.py`
  current best risk-controlled fallback from meta model to latent model

- `hist_residual_family_gate.py`
  HistGradientBoosting residual expert plus family router

- `stacked_family_gate.py`
  current best stacked family router over post-processing experts

- `isotonic_family_model.py`
  current best structural residual model using monotone selectivity curves

- `isotonic_stack_gate.py`
  conservative table-level gate that reproduces `isotonic_stack_gate`

- `value_residual_target_encoding.py`
  cross-fitted value-level residual target encoding on top of
  `isotonic_stack_gate`

- `value_residual_on_isotonic_family.py`
  same value-level residual idea on top of the online-validated
  `submission_isotonic_family.csv`

- `join_distribution_estimator.py`
  useful reference for physical/star-join assumptions

Supporting artifacts:

- `final_oof_predictions.csv`
- `group_expert_oof.csv`
- `hierarchical_template_oof.csv`
- `grid_surface_oof.csv`
- `lowrank_latent_oof.csv`
- `latent_final_blend_oof.csv`

Most relevant submission files:

- `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`
- `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`
- `submission_shape_surface_m50_d2_a100_s0p3_g0p0.csv`
- `submission_value_te_capped_pair02_on_isotonic.csv`
- `submission_value_te_column_capped_on_isotonic.csv`
- `submission_value_te_id_heavy_on_isotonic.csv`
- `submission_isotonic_family.csv`
- `submission_latent_final_blend.csv`
- `submission_lowrank_latent.csv`
- `submission_rewrite_angle_affine.csv`
- `submission_grid_surface.csv`
- `submission_hierarchical_template.csv`

## Caveats

- `lowrank_latent_v2.py` exists but timed out before producing a result.
- `latent_join_inversion.py` and `fast_latent_join.py` are conceptual attempts that did not finish successfully in this environment.
- Some older submissions remain in the folder but are no longer competitive relative to the later files listed above.

## Short Recommendation

If someone picks this up cold:

- current online best is `submission_value_te_capped_pair02_on_isotonic.csv`
  with score `3.15321`
- next submit `submission_shape_surface_m50_d2_a10_s0p3_g0p01.csv`
- if that improves, try `submission_shape_surface_m50_d2_a10_s0p3_g0p0.csv`
- if shape surfaces do not transfer, the next larger jump likely requires
  obtaining the original JOB/IMDb tables and doing exact or near-exact counts
