# Heart (Cleveland) Synthetic Generation

- **Source**: `TabPFN/sync_data_proj/datasets/EHR_datasets/heart/dataset.csv`
- **Label**: `label_heart_disease`
- **Features & types**  
  - Integer categorical: `sex`, `chest_pain_type`, `fasting_blood_sugar`, `resting_ecg`, `exercise_induced_angina`, `st_slope`, `num_major_vessels`, `thalassemia`  
  - Integer continuous: `age`, `resting_bp`, `cholesterol`, `max_heart_rate`  
  - Float: `st_depression`
- **Cleaning**: convert `?`→NaN; numeric coercion; fill integers with median (continuous) or mode (categorical), fill float with median; round and cast integer columns; label cast to int; clip synthetic features to the real data min/max per column.
- **Generator**: `TabPFGen` with `n_sgld_steps=800`, default `sgld_step_size/sgld_noise_scale`, device=`cuda`, seeds=42. Balanced initialization per class to hit target sample count, SGLD updates, then inverse-scale to original space.
- **Outputs**:
  - `EHR_datasets/heart/synthetic_100.csv` — 303 rows (labels: 0=152, 1=151)
  - `EHR_datasets/heart/synthetic_200.csv` — 606 rows (labels: 0=303, 1=303)
  - `EHR_datasets/heart/synthetic_1000.csv` — 3030 rows (labels: 0=1515, 1=1515)

# Arrhythmia Synthetic Generation

- **Source**: `TabPFN/sync_data_proj/datasets/EHR_datasets/arrhythmia/dataset.csv`
- **Label**: `label_arrhythmia_class`
- **Cleaning**: parse `?` as NaN; cast all features to numeric; label cast to `Int64`; cleaned file saved to `EHR_datasets/arrhythmia/dataset_clean.csv` for reproducibility.
- **Generator**: `TabPFNConditionalGenerator` (TabPFN disabled to force sklearn fallback HistGradientBoosting models), config: `sample_count=452`, `seed=42`, `num_gibbs_rounds=3`, `batch_size=256`, `clip_quantile_low/high=0.01/0.99`, `use_gpu=False`, `target_column=label_arrhythmia_class`, `preserve_label_distribution=True`, `integer_columns=[label_arrhythmia_class]`.
- **Output**:
  - `EHR_datasets/arrhythmia/sync_tabpfn.csv` — 452 rows; label counts mirror the real data (1:245, 2:44, 3:15, 4:15, 5:13, 6:25, 7:3, 8:2, 9:9, 10:50, 14:4, 15:5, 16:22)

## Arrhythmia — SOTA Synthesizers (from sotas/sota_install.md)

- **SDV GaussianCopula**: `sdv.GaussianCopulaSynthesizer` (`epochs` NA, default copula); discrete `label_arrhythmia_class`; CPU. Output `EHR_datasets/arrhythmia/sync_gaussian_copula.csv` (452×280, labels {1:231, 2:62, 3:17, 4:9, 5:15, 6:26, 7:3, 8:1, 9:10, 10:42, 14:4, 15:3, 16:29}).
- **SDV CTGAN**: `sdv.CTGANSynthesizer` with `epochs=50`, `batch_size=64`, `pac=1`, `cuda=False`; same preprocessing as above. Output `EHR_datasets/arrhythmia/sync_sdv_ctgan.csv` (452×280, labels {1:277, 2:35, 3:14, 4:12, 5:12, 6:16, 7:1, 9:17, 10:48, 14:2, 15:1, 16:17}).
- **SDV TVAE**: `sdv.TVAESynthesizer` with `epochs=50`, `batch_size=64`, `cuda=False`; same preprocessing. Output `EHR_datasets/arrhythmia/sync_sdv_tvae.csv` (452×280, labels heavily collapsed to class 1:451, class 10:1).
- **YData (ydata-synthetic) CTGAN**: `CTGAN` (`ModelParameters` `batch_size=64`, `latent_dim=64`, `layers_dim=128`, `pac=1`, `lr=2e-4`, `betas=(0.5,0.9)`; `TrainParameters` `epochs=10`); ran on CPU (TensorFlow). Output `EHR_datasets/arrhythmia/sync_ydata_ctgan.csv` (452×280, labels {1:112, 2:55, 3:37, 4:23, 5:28, 6:43, 7:14, 8:7, 9:29, 10:37, 14:11, 15:19, 16:37}).
- **SynthCity DDPM (TabDDPM plugin)**: Patched `torch.nn.RMSNorm` for torch 2.2; plugin args `n_iter=120`, `batch_size=128`, `random_state=42`, `device=cpu`; warnings about missing `dgl` ignored. Output `EHR_datasets/arrhythmia/sync_synthcity_ddpm.csv` (452×280, labels collapsed: {1:236, 16:216}).
- **Not completed (documented failures)**:
  - `ydata-sdk`: pip install failed (`psycopg2` build requires pg_config/root privileges); generation skipped.
  - `synthcity_great` / **GReaT**: ran into GPU assertion and sequence length >1024 for distilgpt2 when tokenizing 280-column rows; training aborted before output.
  - `be-great` (standalone) not run because the underlying GReaT path above failed on this wide table; would need column pruning/shorter tokenization to fit GPT2 context.
  - `ForestDiffusion`: repeated attempts with very small `n_t`/`n_estimators` and label remapping still timed out on 279-feature input; no output written.
