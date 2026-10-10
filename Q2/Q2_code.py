#Imports
from pyspark.sql import SparkSession 
from pyspark.sql import functions as F 
from pyspark.sql.functions import month, weekday, col 
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import csv
import logging


#Spark Builder
spark = SparkSession.builder \
    .config("spark.local.dir","/mnt/parscratch/users/acp25ra") \
    .getOrCreate()


#Section A
spark.sparkContext.setLogLevel("ERROR")
logging.getLogger("py4j").setLevel(logging.ERROR)
spark._jvm.org.apache.log4j.LogManager.getLogger("org.apache.spark.ml.util.Instrumentation") \
    .setLevel(spark._jvm.org.apache.log4j.Level.ERROR)



#Load Dataset

DATA_PATH = "/mnt/parscratch/users/com6012_2026/data/dft_traffic_counts_raw_counts.csv"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("nullValue","NULL")
    .csv(DATA_PATH)
)

#Extract Month and Weekday
df = df.withColumn("month", month(col("count_date"))) \
       .withColumn("weekday", weekday(col("count_date")))

df = df.withColumn("direction_of_travel", F.upper(col("direction_of_travel")))

df = df.dropna(subset=["all_motor_vehicles"])

#one-hot encode
categorical_cols = [
    "direction_of_travel", "hour", "region_ons_code",
    "local_authority_code", "month", "weekday" ]


#convert categories to numeric indices
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")

    for c in categorical_cols
]

# OneHotEncoder
encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")

    for c in categorical_cols 
]

pipeline = Pipeline(stages=indexers + encoders) 
df = pipeline.fit(df).transform(df) 

#Data types
df = df.withColumn("latitude", col("latitude").cast("double")) \
        .withColumn("longitude", col("longitude").cast("double")) \
         .withColumn("all_motor_vehicles", col("all_motor_vehicles").cast("double"))


lat_lon_assembler = VectorAssembler(
    inputCols=["latitude", "longitude"],
    outputCol="lat_lon_vec" )
df = lat_lon_assembler.transform(df)

# Standardize
scaler = StandardScaler(
    inputCol="lat_lon_vec",
    outputCol="lat_lon_scaled",
    withMean=True,
    withStd=True )
df = scaler.fit(df).transform(df)

#split by year

train_df = df.filter(col("year") <= 2021).cache() 
val_df = df.filter((col("year") >= 2022) & (col("year") <= 2023)).cache() 
test_df = df.filter(col("year") == 2024).cache()


#Print the result of processed data
print("\nSample of processed columns (3 rows):") 
df.select(
    "count_date", "month", "weekday",
    "direction_of_travel",
    "all_motor_vehicles",
    "lat_lon_scaled"
).show(3, truncate=False)


n_train = train_df.count() 
n_val = val_df.count() 
n_test = test_df.count() 

print("\nData split by year") 
print(f"Training   (2000-2021): {n_train:,} rows") 
print(f"Validation (2022-2023): {n_val:,} rows") 
print(f"Testing    (2024):      {n_test:,} rows")
print(f"Total:                  {n_train + n_val + n_test:,} rows")


# Section B: Poisson Regression
#My Registration Number: 250117976
#Last 5 digits used for seed: 17976

feature_cols = [f"{c}_ohe" for c in categorical_cols] + ["lat_lon_scaled"]

final_assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

train_df = final_assembler.transform(train_df).cache()
val_df   = final_assembler.transform(val_df).cache()
test_df  = final_assembler.transform(test_df).cache()

print("\nFinal feature vector created")
print(f"Feature columns combined: {feature_cols}")
train_df.select("features", "all_motor_vehicles").show(3, truncate=80)


#Hyperparameter tuning 

reg_params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

seeds = [17976, 17977, 17978, 17979, 17980]

# Evaluator for MSE on validation set
evaluator = RegressionEvaluator(
    labelCol="all_motor_vehicles",
    predictionCol="prediction",
    metricName="mse"
)

# Store results:
results = {rp: [] for rp in reg_params}

print("\nHyperparameter tuning started")
#Hyperparameter tuning

for rp in reg_params:
    for seed in seeds:
        sampled_train = train_df.sample(withReplacement=False, fraction=0.8, seed=seed)

        glr = GeneralizedLinearRegression(
            family="poisson",
            link="log",
            featuresCol="features",
            labelCol="all_motor_vehicles",
            regParam=rp,
            maxIter=25
        )

        model = glr.fit(sampled_train)
        predictions = model.transform(val_df)
	#Evaluation
        mse = evaluator.evaluate(predictions)
        results[rp].append(mse)

        print(f"regParam={rp}, seed={seed}, val_MSE={mse:.4f}")

summary = []
for rp in reg_params:
    mses = results[rp]
    mean_mse = float(np.mean(mses))
    std_mse  = float(np.std(mses))
    summary.append((rp, mean_mse, std_mse))
    print(f"regParam={rp}: mean_MSE={mean_mse:.4f}, std_MSE={std_mse:.4f}")

# Save results to CSV
with open("Q2_B_validation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["regParam", "mean_val_MSE", "std_val_MSE"])
    for rp, m, s in summary:
        writer.writerow([rp, m, s])

with open("Q2_B_validation_raw.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["regParam", "seed", "val_MSE"])
    for rp in reg_params:
        for seed, mse in zip(seeds, results[rp]):
            writer.writerow([rp, seed, mse])


print("\nTuning complete. Results saved to Q2_B_validation_results.csv")




rp_values = [s[0] for s in summary]
mean_vals = [s[1] for s in summary]
std_vals  = [s[2] for s in summary]

plt.figure(figsize=(8, 5))
plt.errorbar(rp_values, mean_vals, yerr=std_vals,
             fmt='-o', capsize=5, color='steelblue',
             ecolor='black', label='Mean val MSE ± std')
plt.xscale('log')
plt.xlabel('regParam (log scale)')
plt.ylabel('Validation MSE')
plt.title('Poisson Regression: Validation MSE vs regParam')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Q2_B_validation_curve.png", dpi=150)
plt.close()
print("\nValidation curve saved to Q2_B_validation_curve.png")

# Pick optimal regParam
optimal_rp = min(summary, key=lambda x: x[1])[0]
print(f"Optimal regParam: {optimal_rp}")

train_val_df = train_df.union(val_df).cache()
print(f"Combined train+val size: {train_val_df.count():,} rows")

final_glr = GeneralizedLinearRegression(
    family="poisson",
    link="log",
    featuresCol="features",
    labelCol="all_motor_vehicles",
    regParam=optimal_rp,
    maxIter=25
)

final_model = final_glr.fit(train_val_df)

# Evaluate on test set
test_predictions = final_model.transform(test_df)
test_mse = evaluator.evaluate(test_predictions)

print(f"\nFinal Poisson Model")
print(f"Optimal regParam: {optimal_rp}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Intercept: {final_model.intercept}")
print(f"Number of coefficients: {len(final_model.coefficients)}")

# Save coefficients 
with open("Q2_B_final_results.txt", "w") as f:
    f.write("=== Section B: Poisson Regression - Final Results ===\n")
    f.write(f"Optimal regParam: {optimal_rp}\n")
    f.write(f"Test MSE: {test_mse:.4f}\n")
    f.write(f"Intercept: {final_model.intercept}\n")
    f.write(f"Number of coefficients: {len(final_model.coefficients)}\n\n")
    f.write("=== Learned Coefficients ===\n")
    for i, coef in enumerate(final_model.coefficients):
        f.write(f"coef[{i}]: {coef}\n")

with open("Q2_B_coefficients.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "coefficient"])
    for i, coef in enumerate(final_model.coefficients):
        writer.writerow([i, float(coef)])


print("\nFinal results saved to Q2_B_final_results.txt")
print("\nSection B complete")



# Section C: Logistic Regression
# Student Registration Number: 250117976
# Last 5 digits used for seeds: 17976

# Compute median
print("\nSection C: Logistic Regression")

median_val = train_df.approxQuantile("all_motor_vehicles", [0.5], 0.001)[0]
print(f"Median of all_motor_vehicles (training set only): {median_val}")

train_df = train_df.withColumn(
    "traffic", (col("all_motor_vehicles") > median_val).cast("double")
).cache()

val_df = val_df.withColumn(
    "traffic", (col("all_motor_vehicles") > median_val).cast("double")
).cache()

test_df = test_df.withColumn(
    "traffic", (col("all_motor_vehicles") > median_val).cast("double")
).cache()


print("\nClass distribution in training set:")
train_df.groupBy("traffic").count().orderBy("traffic").show()

print("Class distribution in validation set:")
val_df.groupBy("traffic").count().orderBy("traffic").show()

print("Class distribution in test set:")
test_df.groupBy("traffic").count().orderBy("traffic").show()

#Hyperparameter tuning

reg_params_lr = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
elastic_params = [0.0, 0.2, 0.5, 0.8, 1.0]

# Seeds
seeds_lr = [17976, 17977, 17978, 17979, 17980]
#Eval
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="traffic",
    predictionCol="prediction",
    metricName="accuracy"
)

# Store results:
results_lr = {(rp, en): [] for rp in reg_params_lr for en in elastic_params}

print("\nLogistic regression hyperparameter tuning started")

for rp in reg_params_lr:
    for en in elastic_params:
        for seed in seeds_lr:
            # Sample 80%
            sampled_train = train_df.sample(withReplacement=False, fraction=0.8, seed=seed)

            lr = LogisticRegression(
                featuresCol="features",
                labelCol="traffic",
                regParam=rp,
                elasticNetParam=en,
                maxIter=25
            )

            model = lr.fit(sampled_train)
            predictions = model.transform(val_df)

            # Compute validation
            acc = acc_evaluator.evaluate(predictions)
            results_lr[(rp, en)].append(acc)

            print(f"regParam={rp}, elasticNet={en}, seed={seed}, val_acc={acc:.4f}")

# Compute mean and std
summary_lr = []
for rp in reg_params_lr:
    for en in elastic_params:
        accs = results_lr[(rp, en)]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        summary_lr.append((rp, en, mean_acc, std_acc))
        print(f"regParam={rp}, elasticNet={en}: mean_acc={mean_acc:.4f}, std_acc={std_acc:.4f}")

# Save results to CSV
with open("Q2_C_validation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["regParam", "elasticNetParam", "mean_val_acc", "std_val_acc"])
    for rp, en, m, s in summary_lr:
        writer.writerow([rp, en, m, s])


with open("Q2_C_validation_raw.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["regParam", "elasticNetParam", "seed", "val_acc"])
    for rp in reg_params_lr:
        for en in elastic_params:
            for seed, acc in zip(seeds_lr, results_lr[(rp, en)]):
                writer.writerow([rp, en, seed, acc])

print("\nTuning complete. Results saved to Q2_C_validation_results.csv")


#Plot validation curves for each elasticNetParam
plt.figure(figsize=(10, 6))

for i, en in enumerate(elastic_params):
    means = [s[2] for s in summary_lr if s[1] == en]
    stds  = [s[3] for s in summary_lr if s[1] == en]
    plt.errorbar(
        reg_params_lr, means, yerr=stds,
        fmt='-o', capsize=4,
        label=f'elasticNetParam={en}'
    )

plt.xscale('log')
plt.xlabel('regParam (log scale)')
plt.ylabel('Validation Accuracy')
plt.title('Logistic Regression: Validation Accuracy vs regParam (per elasticNetParam)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("Q2_C_validation_curve.png", dpi=150)
plt.close()
print("\nValidation curve saved to Q2_C_validation_curve.png")

#Pick optimal
optimal_combo = max(summary_lr, key=lambda x: x[2])
optimal_rp_lr  = optimal_combo[0]
optimal_en_lr  = optimal_combo[1]
optimal_acc    = optimal_combo[2]
print(f"Optimal regParam: {optimal_rp_lr}")
print(f"Optimal elasticNetParam: {optimal_en_lr}")
print(f"Best mean validation accuracy: {optimal_acc:.4f}")

#Retrain on combined train + validation sets
train_val_lr_df = train_df.union(val_df).cache()
print(f"Combined train+val size for LR: {train_val_lr_df.count():,} rows")

final_lr = LogisticRegression(
    featuresCol="features",
    labelCol="traffic",
    regParam=optimal_rp_lr,
    elasticNetParam=optimal_en_lr,
    maxIter=25
)

final_lr_model = final_lr.fit(train_val_lr_df)

# Evaluate on test set
test_predictions_lr = final_lr_model.transform(test_df)
test_acc = acc_evaluator.evaluate(test_predictions_lr)

print(f"\nFinal Logistic Regression Model")
print(f"Optimal regParam: {optimal_rp_lr}")
print(f"Optimal elasticNetParam: {optimal_en_lr}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Intercept: {final_lr_model.intercept}")
print(f"Number of coefficients: {len(final_lr_model.coefficients)}")

#Save final results to file
with open("Q2_C_final_results.txt", "w") as f:
    f.write("=== Section C: Logistic Regression - Final Results ===\n")
    f.write(f"Optimal regParam: {optimal_rp_lr}\n")
    f.write(f"Optimal elasticNetParam: {optimal_en_lr}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Intercept: {final_lr_model.intercept}\n")
    f.write(f"Number of coefficients: {len(final_lr_model.coefficients)}\n\n")
    f.write("=== Learned Coefficients ===\n")
    for i, coef in enumerate(final_lr_model.coefficients):
        f.write(f"coef[{i}]: {coef}\n")

# coefficients as CSV OPTIONAL
with open("Q2_C_coefficients.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "coefficient"])
    for i, coef in enumerate(final_lr_model.coefficients):
        writer.writerow([i, float(coef)])

print("\nFinal results saved to Q2_C_final_results.txt and Q2_C_coefficients.csv")
print("\nSection C complete")
