# Computational Environment
Generated: 2026-08-05 08:36:50 UTC

## 1. Hardware and OS

| Item             | Value                                               |
|------------------|-----------------------------------------------------|
| OS               | Windows-10-10.0.26200-SP0                           |
| CPU              | Intel64 Family 6 Model 151 Stepping 2, GenuineIntel |
| Logical cores    | 16                                                  |
| RAM (GB)         | 31.9                                                |
| Architecture     | AMD64                                               |
| GPU (nvidia-smi) | NVIDIA GeForce RTX 3060 Ti, 591.86, 8192 MiB        |

## 2. Interpreter and environment manager

| Item       | Value                                            |
|------------|--------------------------------------------------|
| Python     | 3.11.5 (CPython, MSC v.1916 64 bit (AMD64))      |
| Executable | C:\ProgramData\anaconda3\envs\pytorch\python.exe |
| Conda env  | pytorch                                          |
| Conda      | conda 23.7.4
[0m                                |
| Virtualenv | no                                               |
| pip        | pip 23.3.1                                       |

## 3. Deep-learning stack

| Item                          | Value        |
|-------------------------------|--------------|
| torch                         | 2.5.1+cu124  |
| torch built with CUDA         | 12.4         |
| cuDNN (torch build)           | 90100        |
| CUDA available at runtime     | True         |
| nvcc (system CUDA toolkit)    | not on PATH  |
| cudnn.deterministic (default) | False        |
| cudnn.benchmark (default)     | False        |
| TF32 allowed (matmul / cudnn) | False / True |
| float32_matmul_precision      | highest      |
| torch intra-op threads        | 10           |

The `torch` wheel must be reinstalled with the matching CUDA build, e.g.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## 4. Numerical backend (BLAS)

Reduction order in BLAS depends on vendor and thread count, so float results can differ in the last digits between machines even with identical package versions.

| Item             | Value                            |
|------------------|----------------------------------|
| numpy            | 1.26.4                           |
| BLAS             | openblas64 0.3.23.dev            |
| LAPACK           | dep2270588361616 1.26.4          |
| openmp threads   | 10 (vNone)                       |
| openmp threads   | 1 (vNone)                        |
| openblas threads | 16 (v0.3.23.dev, layer=pthreads) |

## 5. Packages

### Tier A — version affects the reported numbers (pin exactly)

| Package      | Version     | Why it matters                                                                 |
|--------------|-------------|--------------------------------------------------------------------------------|
| captum       | 0.7.0       | Integrated Gradients attributions in LooMen_2_explain                          |
| joblib       | 1.3.2       | Parallel() chunking in Figure1_1_run                                           |
| llvmlite     | 0.42.0      | numba backend                                                                  |
| numba        | 0.59.1      | compiles UMAP internals; different numba => different embedding                |
| numpy        | 1.26.4      | core numerics; NumPy 2.x changed dtype promotion and repr                      |
| pandas       | 2.3.3       | groupby/merge semantics, NA handling, CSV parsing                              |
| polars       | 1.35.2      | parquet reading and aggregations in 4_Cluster                                  |
| pynndescent  | 0.5.12      | UMAP kNN graph construction                                                    |
| scikit-learn | 1.8.0       | metrics (roc_auc/f1) and train_test_split shuffling                            |
| scipy        | 1.17.1      | stats tests (mannwhitneyu/linregress); implementations changed across versions |
| statsmodels  | 0.14.2      | multipletests FDR correction in 5_City/4_Scaling                               |
| tokenizers   | 0.21.4      | actual tokenization; can shift token ids across versions                       |
| torch        | 2.5.1+cu124 | model training/inference; must be pinned WITH its CUDA build                   |
| transformers | 4.51.0      | Trainer loop, scheduler, tokenizer behaviour                                   |
| umap-learn   | 0.5.11      | UMAP embedding in 5_City/1_UMAP; NOT stable across versions                    |

### Tier B — version affects data loading / model loading

| Package         | Version | Why it matters                                                   |
|-----------------|---------|------------------------------------------------------------------|
| accelerate      | 1.2.1   | required by transformers.Trainer; controls fp16/device placement |
| datasets        | 4.4.2   | optional Trainer dependency                                      |
| fiona           | 1.10.1  | vector IO (older geopandas)                                      |
| geopandas       | 1.0.1   | geojson reading in 5_City/2_Map                                  |
| huggingface-hub | 0.36.0  | checkpoint download/resolution                                   |
| pillow          | 10.0.1  | image mask in 4_Cluster/1_Cluster_round                          |
| pyarrow         | 22.0.0  | parquet backend                                                  |
| pyogrio         | 0.10.0  | vector IO (newer geopandas default)                              |
| pyproj          | 3.7.1   | CRS transforms                                                   |
| safetensors     | 0.4.5   | checkpoint format                                                |
| shapely         | 2.1.0   | geometry ops; 1.x vs 2.x is a breaking change                    |
| tqdm            | 4.67.1  | progress bars only                                               |

### Tier C — version affects figure appearance only

| Package    | Version | Why it matters                                            |
|------------|---------|-----------------------------------------------------------|
| cartopy    | 0.24.1  | map projection and coastlines in 5_City/2_Map             |
| matplotlib | 3.8.2   | all figures; default style changes between minor versions |
| wordcloud  | 1.9.3   | wordcloud figure; layout depends on version + font        |

### Unclassified (detected in code, not in the curated registry)

| Package       | Version | Why it matters |
|---------------|---------|----------------|
| threadpoolctl | 3.2.0   | —              |

### Geospatial C libraries (not captured by pip)

| Library          | Version                                                                            |
|------------------|------------------------------------------------------------------------------------|
| shapely          | 2.1.0                                                                              |
| geos             | 3.13.1                                                                             |
| pyproj           | 3.7.1                                                                              |
| proj             | 9.5.1                                                                              |
| proj_data_dir    | C:\ProgramData\anaconda3\envs\pytorch\Lib\site-packages\pyproj\proj_dir\share\proj |
| fiona            | 1.10.1                                                                             |
| gdal_via_fiona   | 3.9.1                                                                              |
| pyogrio          | 0.10.0                                                                             |
| gdal_via_pyogrio | (3, 9, 1)                                                                          |

## 6. Fonts

The plotting scripts hard-code font families. A missing font does **not** raise an error — matplotlib silently substitutes DejaVu Sans and the figure changes.

| Font            | Status    |
|-----------------|-----------|
| Arial           | available |
| Helvetica       | MISSING   |
| DejaVu Sans     | available |
| Microsoft YaHei | available |
| SimHei          | available |
| Times New Roman | available |

> Missing on this machine: Helvetica

## 7. Model checkpoints

Paths found in the code. Pin the exact checkpoint (Hugging Face repo id **and** commit revision), not just the `transformers` version — the same repo id can point to different weights over time.

| Checkpoint / `hfl/chinese-roberta-wwm-ext`                | Referenced in                                                      |
|----------------------------------|--------------------------------------------------------------------|
| D:/Model/chinese-roberta-wwm-ext | Code\3_LooMen\LooMen_1_train.py, Code\3_LooMen\LooMen_2_explain.py |

## 8. Environment variables in effect

| Variable          | Value                                 |
|-------------------|---------------------------------------|
| CONDA_DEFAULT_ENV | pytorch                               |
| CONDA_PREFIX      | C:\ProgramData\anaconda3\envs\pytorch |

## 9. Rebuilding this environment

conda create -n repro python=3.11
conda activate repro
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt