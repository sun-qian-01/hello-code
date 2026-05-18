# Cardinality Estimation Handoff

## Goal

This folder contains a long-running attempt to improve leaderboard score for the SQL cardinality estimation task.

The working objective is:

- Input: simplified SQL query fields from `train.csv` / `test.csv`
- Output: predicted `Cardinality`
- Metric: Mean Q-Error

The user reported an online score of about `4.88061` on the original submission, then iterated through many experimental routes. The best local OOF result reached so far is about `3.67916`, but online gains were much smaller than local improvements, so overfitting to local OOF is a real risk.

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

1. `submission_meta_gate_huber_resid.csv`
   Current best local OOF after a tiny residual meta-model:
   about `3.67916`

2. `submission_column_family_gate_shape50_cols80_affine.csv`
   Conservative affine-calibrated gate:
   about `3.67961`

3. `submission_column_family_gate_shape50_cols80.csv`
   Best uncalibrated nested-gate OOF:
   about `3.67994`

4. `submission_column_family_gate_shape_cols.csv`
   Slightly more conservative two-level gate:
   about `3.68027`

5. `submission_column_family_gate.csv`
   Column-family gate with wider expert pool:
   about `3.68062`

6. `submission_column_family_gate_core.csv`
   Conservative column-family gate using only strongest late-stage experts:
   about `3.68616`

7. `submission_latent_final_blend.csv`
   Previous best local OOF:
   about `3.69370`

8. `submission_lowrank_latent.csv`
   Low-rank latent residual model only:
   about `3.69431`

9. `submission_rewrite_angle_affine.csv`
   Best pre-latent template-first route:
   about `3.70057`

10. `submission_grid_surface.csv`
   Template surface reconstruction route:
   about `3.70169`

All of these files were checked for:

- exactly 5,000 rows
- columns `Id,Cardinality`
- unique IDs
- positive integer predictions

## Recommended Starting Point

If continuing immediately, start from:

- `submission_meta_gate_huber_resid.csv`
- `meta_gate_residual.py`
- `column_family_gate.py`
- `lowrank_latent_residual.py`
- `hierarchical_template_model.py`
- `HANDOFF.md`

That combination represents the strongest current route plus the best previous fallback.
Because online gains have lagged local OOF gains before, also keep
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

Important:

- online scores improved much less than local OOF
- small OOF improvements often translated into only ~0.01 online
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

## What To Try Next

If someone continues this work, the best next steps are:

1. Push the low-rank latent route further, but efficiently.
   Use smaller/faster SVD sweeps and better residual experts rather than direct Python simulators.

2. Move from global latent factors to per-family latent factors.
   Separate latent spaces for:
   - `title + cast_info`
   - `title + movie_companies`
   - `title + movie_info`
   - `title + movie_keyword`
   and the 3-table combinations built from them.

3. Use stronger interaction tokenization.
   Especially:
   - pairwise predicate-bin interactions
   - table-specific value-bin crosses
   - shape-conditioned latent features

4. If external data ever becomes available, reconstruct actual table-level distributions.
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

- submit `submission_meta_gate_huber_resid.csv` first
- compare online result to `submission_column_family_gate_shape50_cols80_affine.csv`,
  `submission_column_family_gate_core.csv`, and `submission_latent_final_blend.csv`
- if online gain is weak again, continue from `lowrank_latent_residual.py`, not from the older global LightGBM scripts
- focus on better latent interaction features, not more generic booster tuning
