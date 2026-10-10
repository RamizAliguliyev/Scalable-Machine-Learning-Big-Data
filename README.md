# Scalable Machine Learning with PySpark

Large-scale data analysis and machine learning with **Apache Spark (PySpark)**, run on the University of Sheffield HPC cluster (Stanage).

The project covers four tasks on four real-world datasets, from log analytics on millions of web requests to distributed model training on 11 million particle-physics events and 20 million movie ratings.

Completed as coursework for **COM6012 Scalable Machine Learning**, University of Sheffield (2026).

---

## Tech Stack

- **Apache Spark / PySpark:** DataFrames, Spark SQL, Spark MLlib
- **MLlib models:** Poisson regression (GLM), Logistic Regression, Random Forest, Gradient-Boosted Trees, ALS, K-Means
- **Tuning:** `ParamGridBuilder`, `CrossValidator`, manual grid search with repeated subsampling
- **Python:** pandas, NumPy, Matplotlib
- **Infrastructure:** University of Sheffield HPC, jobs submitted with SLURM (`sbatch`)

## Overview

| Task | Dataset | Size | Techniques |
|------|---------|------|------------|
| **Q1** Log mining | NASA HTTP access logs (July 1995) | ~1.9M requests | Regex parsing, aggregation, heatmaps |
| **Q2** Regression & classification | UK DfT road traffic counts | ~5.1M rows | One-hot encoding, scaling, Poisson GLM, Logistic Regression |
| **Q3** Tree ensembles | HIGGS (UCI) | 11M events | Random Forest vs GBT, 3-fold cross-validation |
| **Q4** Recommender systems | MovieLens 20M | 20M ratings | ALS collaborative filtering, K-Means user clustering |

---

## Q1: NASA Log Analysis

Parses raw web-server logs and analyses traffic from UK academic (`.ac.uk`), company (`.co.uk`) and government (`.gov.uk`) hosts.

- Counted total requests and unique hosts per sector
- Extracted institution domains (e.g. `www.shef.ac.uk` → `shef.ac.uk`) and ranked them
- Compared the top 9 UK companies against all others
- Built day × hour heatmaps of request activity

| Sector | Total requests | Unique hosts |
|--------|---------------:|-------------:|
| Academic (`.ac.uk`) | 25,012 | 1,023 |
| Companies (`.co.uk`) | 22,503 | 1,118 |
| Government (`.gov.uk`) | 195 | 15 |

**Key finding:** `demon.co.uk` (Demon Internet, a major 1995 UK ISP) generated about 52% of all UK company traffic, more than all other companies outside the top 9 combined.

## Q2: Traffic Prediction with Generalised Linear Models

Predicts hourly motor-vehicle counts on UK roads.

- **Preprocessing:** extracted month and weekday, normalised categories, dropped nulls, one-hot encoded 6 categorical features, standardised latitude/longitude
- **Time-based split:** train 2000–2021 (4.66M rows), validation 2022–2023 (307K), test 2024 (152K)
- **Tuning:** 7 values of `regParam` (plus 5 values of `elasticNetParam` for LR), each evaluated on 5 random 80% subsamples, reported as mean ± std

| Model | Best hyperparameters | Test result |
|-------|----------------------|-------------|
| Poisson regression (log link) | `regParam = 0.001` | MSE = 787,575 |
| Logistic regression (above/below median traffic) | `regParam = 0.001`, `elasticNetParam = 0.0` (L2) | Accuracy = 64.12% |

**Key finding:** both models performed best with the weakest regularisation. Strong L1 / elastic-net penalties removed informative one-hot features and caused underfitting.

## Q3: Random Forest vs Gradient-Boosted Trees on HIGGS

Binary classification of signal vs background events from 28 physics features.

1. Tuned both models with 3-fold cross-validation on a **2% stratified sample** (220K rows), using a 27-combination grid each
2. Retrained the best configurations on the **full 11M-row dataset**

| Model | Best parameters | 2% sample test acc. | Full dataset test acc. |
|-------|-----------------|--------------------:|-----------------------:|
| Random Forest | `numTrees=150`, `maxDepth=12`, `minInstancesPerNode=5` | 71.67% | 71.76% |
| GBT | `maxIter=30`, `maxDepth=5`, `stepSize=0.2` | 71.92% | **71.95%** |

**Key finding:** GBT was marginally better. On the full dataset train and test accuracy were almost identical for both models, so there was no clear overfitting.

## Q4: Movie Recommendations and User Clustering

**Part A: ALS with time-based splits.** Ratings were split chronologically (40/60, 60/40, 80/20) and two ALS settings were compared.

| Split | Setting 1 (rank 10, 10 iter) RMSE | Setting 2 (rank 20, 15 iter) RMSE |
|-------|----------------------------------:|----------------------------------:|
| 40/60 | 0.8021 | 0.8006 |
| 60/40 | 0.7753 | **0.7738** |
| 80/20 | 0.7951 | 0.7947 |

**Part B: User analysis.** K-Means (k = 25) was applied to the learned ALS user factors. For the largest cluster in each split, the most common genres among movies rated ≥ 4.0 were found. Drama ranked first in every split, followed by Comedy and Romance.

**Key finding:** the 60/40 split beat 80/20, likely because the most recent period contains newer users and films (concept drift). More training data is not always better under realistic time-based evaluation.

---

## Repository Structure

```
├── Q1_code.py      # NASA log analysis
├── Q2_code.py      # Poisson & logistic regression on traffic data
├── Q3_code.py      # Random Forest vs GBT on HIGGS
├── Q4_code.py      # ALS recommendations & K-Means clustering
├── <report>.pdf    # Full report with figures and discussion
└── README.md
```

## Datasets

The datasets are **not included** because of their size. Download them and place them in a `Data/` folder:

| Task | Dataset | Source |
|------|---------|--------|
| Q1 | `NASA_access_log_Jul95` | [NASA HTTP logs (Internet Traffic Archive)](https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html) |
| Q2 | `dft_traffic_counts_raw_counts.csv` | [UK DfT road traffic statistics](https://roadtraffic.dft.gov.uk/downloads) |
| Q3 | `HIGGS.csv` | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/280/higgs) |
| Q4 | `ml-20m/ratings.csv`, `ml-20m/movies.csv` | [GroupLens MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) |

## How to Run

Requirements: Python 3, Java 8+/11, PySpark 3.x, pandas, NumPy, Matplotlib.

```bash
pip install pyspark pandas numpy matplotlib
```

Run a task locally:

```bash
spark-submit Q1_code.py
```

On the Sheffield HPC each task was submitted as a SLURM batch job:

```bash
sbatch Q1_script.sh
```

**Note:** some scripts contain HPC-specific paths (e.g. `spark.local.dir` pointing to `/mnt/parscratch/...` and the Q2 dataset path). Update these to your own directories before running elsewhere. Q3 and Q4 train on 11M–20M rows and need substantial memory and time.

## Outputs

Each script writes its results to the working directory:

- **Q1:** `Q1_table1.csv`, `Q1_BarChart.png`, three heatmaps
- **Q2:** validation CSVs, validation curves, final results and coefficients for both models
- **Q3:** results printed to stdout
- **Q4:** `Q4_A_metrics.csv/.png`, `Q4_B_cluster_sizes.csv/.png`, `Q4_B_top_genres.csv`

## Author

**Ramiz Aliguliyev**, MSc, University of Sheffield
