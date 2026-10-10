# Registration Number: 250117976

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector


SEED = 17976

ratings_path = "Data/ml-20m/ratings.csv"
movies_path = "Data/ml-20m/movies.csv"

split_fractions = [0.40, 0.60, 0.80]
k_clusters = 25
top_n_clusters = 5
top_n_genres = 10
rating_threshold = 4.0


spark = SparkSession.builder.appName("Q4").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Q4 - Movie Recommendations and Cluster Analysis")
print("Seed:", SEED)


#PART A
print("\nTime-split Recommendations\n")

ratings = spark.read.option("header", True).option("inferSchema", True).csv(ratings_path)
ratings = ratings.select("userId", "movieId", "rating", "timestamp").cache()

n_ratings = ratings.count()
print("Total number of ratings:", n_ratings)


#timestamp thresholds for the three splits
print("\nComputing timestamp thresholds")
thresholds = ratings.approxQuantile("timestamp", split_fractions, 0.0001)
for frac, thr in zip(split_fractions, thresholds):
    print("  ", int(frac*100), "% percentile timestamp =", thr)


splits = {}
print("\nSplit sizes:")
for frac, thr in zip(split_fractions, thresholds):
    train = ratings.filter(F.col("timestamp") <= thr).cache()
    test = ratings.filter(F.col("timestamp") > thr).cache()
    pct = int(frac * 100)
    print("  Split", str(pct) + "/" + str(100-pct),
          ": train =", train.count(), "  test =", test.count())
    splits[frac] = (train, test)


evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

#ALS hyperparameter tuning (Setting 1: default, Setting 2: rank=20, maxIter=15)
results = []
setting2_models = {}

for frac in split_fractions:
    train_df, test_df = splits[frac]
    pct = int(frac * 100)
    split_label = str(pct) + "/" + str(100-pct)
    print("\nSplit", split_label)

    for name in ["Setting1", "Setting2"]:
        print("  Training ALS", name, "...")
        t0 = time.time()

        if name == "Setting1":

            als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
                      coldStartStrategy="drop", seed=SEED)
        else:
            # Setting 2: rank=20, maxIter=15. More latent factors and more iterations
	    #with 20M ratings,
	    #rank = 10 likely underfits user/movie diversity; doubling rank
	    #to 20 lets ALS capture finer-grained latent structure, and lifting
	    #maxIter to 15 gives the larger model more rounds to overlap.  This
	    #is a safier direction for potential improvement, while keeping
	    #regParam at the default 0.1 so the change is attributable to model
	    #capacity rather than regularisation.
            als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
                      rank=20, maxIter=15, regParam=0.1,
                      coldStartStrategy="drop", seed=SEED)

        model = als.fit(train_df)
        fit_time = time.time() - t0

        preds = model.transform(test_df).cache()
        preds.count()

        rmse = evaluator_rmse.evaluate(preds)
        mse = evaluator_mse.evaluate(preds)
        mae = evaluator_mae.evaluate(preds)
        preds.unpersist()

        print("   ", name, ": fit =", round(fit_time, 1), "s",
              "  RMSE =", round(rmse, 4),
              "  MSE =", round(mse, 4),
              "  MAE =", round(mae, 4))

        results.append({
            "split": split_label,
            "train_pct": pct,
            "setting": name,
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae,
        })

        if name == "Setting2":
            setting2_models[frac] = model


# save results
df_results = pd.DataFrame(results)
df_results.to_csv("Q4_A_metrics.csv", index=False)
print("\nMetrics written to: Q4_A_metrics.csv")


print("\nSummary table (3 splits x 2 settings x 3 metrics = 18 numbers):")
print("Split     Setting     RMSE      MSE       MAE")
for r in results:
    print(r["split"], "  ", r["setting"], "  ",
          round(r["RMSE"], 4), "  ",
          round(r["MSE"], 4), "  ",
          round(r["MAE"], 4))


# figure 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metric_names = ["RMSE", "MSE", "MAE"]
splits_labels = [str(int(f*100)) + "/" + str(100-int(f*100)) for f in split_fractions]
bar_width = 0.35
x_idx = np.arange(len(split_fractions))

for ax, metric in zip(axes, metric_names):
    s1_vals = [r[metric] for r in results if r["setting"] == "Setting1"]
    s2_vals = [r[metric] for r in results if r["setting"] == "Setting2"]
    bars1 = ax.bar(x_idx - bar_width/2, s1_vals, bar_width,
                   label="Setting 1 (rank=10, maxIter=10)")
    bars2 = ax.bar(x_idx + bar_width/2, s2_vals, bar_width,
                   label="Setting 2 (rank=20, maxIter=15)")
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                str(round(b.get_height(), 3)),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(splits_labels)
    ax.set_xlabel("Train / Test split (%)")
    ax.set_ylabel(metric)
    ax.set_title("Test " + metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Q4-A: Time-split ALS evaluation on MovieLens 20M", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig("Q4_A_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Figure written to: Q4_A_metrics.png")


#User Analysis
print("\nUser Analysis\n")

print("Loading movies from:", movies_path)
movies = spark.read.option("header", True).option("inferSchema", True).csv(movies_path)
movies = movies.select("movieId", "title", "genres").cache()
print("Total number of movies:", movies.count())


cluster_size_rows = []
genre_rows = []

for frac in split_fractions:
    pct = int(frac * 100)
    split_label = str(pct) + "/" + str(100-pct)
    print("\nSplit: ", split_label, "")

    als_model = setting2_models[frac]
    train_df, _ = splits[frac]

    # ALS userFactors 
    user_factors = als_model.userFactors
    user_factors_vec = user_factors.withColumn("features",
                                               array_to_vector(F.col("features"))).cache()

    n_users = user_factors_vec.count()
    print("  Users with learned factors:", n_users)

    print("  Fitting k-means (k=" + str(k_clusters) + ", seed=" + str(SEED) + ")...")
    t0 = time.time()
    kmeans = KMeans(k=k_clusters, seed=SEED, featuresCol="features", predictionCol="cluster")
    km_model = kmeans.fit(user_factors_vec)
    print("  k-means fit time:", round(time.time() - t0, 1), "s")

    user_clusters = km_model.transform(user_factors_vec)
    user_clusters = user_clusters.select(F.col("id").alias("userId"), "cluster").cache()
    user_clusters.count()


    # top 5 largest clusters
    cluster_sizes = user_clusters.groupBy("cluster").count().orderBy(F.desc("count"))
    top5 = cluster_sizes.limit(top_n_clusters).collect()

    print("  Top", top_n_clusters, "largest clusters:")
    for rank, row in enumerate(top5, start=1):
        print("    rank", rank, ": cluster id =", row["cluster"], "  size =", row["count"])
        cluster_size_rows.append({
            "split": split_label,
            "cluster_rank": rank,
            "cluster_id": int(row["cluster"]),
            "size": int(row["count"]),
        })

    largest_cluster_id = int(top5[0]["cluster"])
    largest_cluster_size = int(top5[0]["count"])
    print("  Largest cluster id =", largest_cluster_id,
          "(size =", largest_cluster_size, ")")


# genre analysis: largest cluster, training set only
    users_in_largest = user_clusters.filter(F.col("cluster") == largest_cluster_id) \
                                    .select("userId")

    cluster_train_ratings = train_df.join(users_in_largest, on="userId", how="inner") \
                                    .select("movieId", "rating")

    movies_largest_cluster = cluster_train_ratings.groupBy("movieId").agg(
        F.avg("rating").alias("avg_rating"),
        F.count("rating").alias("n_ratings")
    ).cache()

    n_movies_largest = movies_largest_cluster.count()
    print("  Movies rated by users in largest cluster:", n_movies_largest)

    top_movies = movies_largest_cluster.filter(F.col("avg_rating") >= rating_threshold)
    n_top_movies = top_movies.count()
    print("  Movies with average rating >=", rating_threshold, ":", n_top_movies)


    top_movie_genres = top_movies.join(movies, on="movieId", how="inner") \
                                 .select("movieId", F.split(F.col("genres"), r"\|").alias("genre_arr")) \
                                 .withColumn("genre", F.explode("genre_arr")) \
                                 .select("movieId", "genre")

    genre_counts = top_movie_genres.groupBy("genre").count() \
                                   .orderBy(F.desc("count")) \
                                   .limit(top_n_genres) \
                                   .collect()

    print("  Top", top_n_genres, "genres among top_movies:")
    for rank, row in enumerate(genre_counts, start=1):
        print("   ", rank, ".", row["genre"], "count =", row["count"])
        genre_rows.append({
            "split": split_label,
            "rank": rank,
            "genre": row["genre"],
            "count": int(row["count"]),
        })

    movies_largest_cluster.unpersist()
    user_clusters.unpersist()
    user_factors_vec.unpersist()


# save tables
df_clusters = pd.DataFrame(cluster_size_rows)
df_clusters.to_csv("Q4_B_cluster_sizes.csv", index=False)
print("\nCluster-size table written to: Q4_B_cluster_sizes.csv")

df_genres = pd.DataFrame(genre_rows)
df_genres.to_csv("Q4_B_top_genres.csv", index=False)
print("Genre table written to: Q4_B_top_genres.csv")


print("\nSummary table (3 splits x 5 clusters = 15 numbers):")
print("Split    Rank   Cluster ID   Size")
for r in cluster_size_rows:
    print(r["split"], "  ", r["cluster_rank"], "  ",
          r["cluster_id"], "  ", r["size"])

print("\nSummary table (3 splits x 10 genres = 30 genres):")
print("Split    Rank   Genre              Count")
for r in genre_rows:
    print(r["split"], "  ", r["rank"], "  ", r["genre"], "  ", r["count"])


#figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ranks = np.arange(1, top_n_clusters + 1)
n_splits = len(split_fractions)
bar_width = 0.25

for i, frac in enumerate(split_fractions):
    pct = int(frac * 100)
    label = "Split " + str(pct) + "/" + str(100-pct)
    sizes = [r["size"] for r in cluster_size_rows
             if r["split"] == str(pct) + "/" + str(100-pct)]
    offset = (i - (n_splits - 1) / 2) * bar_width
    bars = ax.bar(ranks + offset, sizes, bar_width, label=label)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                str(int(b.get_height())),
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(ranks)
ax.set_xticklabels(["Rank " + str(r) for r in ranks])
ax.set_xlabel("Cluster rank (by size, within split)")
ax.set_ylabel("Number of users")
ax.set_title("Q4-B1: Top-" + str(top_n_clusters) + " largest k-means clusters of ALS Setting-2 user factors")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("Q4_B_cluster_sizes.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nFigure written to: Q4_B_cluster_sizes.png")

print("\nQ4 finished.")
spark.stop()
