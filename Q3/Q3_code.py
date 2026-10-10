# Registration Number: 250117976

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("acp25ra_Q3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load Data
data_path = "Data/HIGGS.csv"

# Defining schema
feature_cols = [f"feature_{i}" for i in range(1, 29)]
schema = StructType([StructField("label", DoubleType(), True)] +
                    [StructField(f, DoubleType(), True) for f in feature_cols])

df = spark.read.csv(data_path, header=False, schema=schema)


# Stratified 2% -Part A
fractions = {0.0: 0.02, 1.0: 0.02}

df_sample = df.sampleBy("label", fractions=fractions, seed=17976)
df_sample.cache()
sample_size = df_sample.count()
print(f"2% stratified sample size: {sample_size}")
print(f"Class distribution in sample:")
df_sample.groupBy("label").count().show()

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Train / Test Split on sample
train_sample, test_sample = df_sample.randomSplit([0.8, 0.2], seed=17976)
train_sample.cache()

print(f"Sample train size: {train_sample.count()}")
print(f"Sample test size: {test_sample.count()}")

#Evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Random Forest hyperparameter tuning (3-fold CV)
print("\n=== Random Forest Cross-Validation ===")

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=17976)

rf_param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees,            [45, 75, 150]) \
    .addGrid(rf.maxDepth,            [5, 8, 12]) \
    .addGrid(rf.minInstancesPerNode, [1, 5, 10]) \
    .build()

rf_pipeline = Pipeline(stages=[assembler, rf])

rf_cv = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=17976,
    parallelism=3
)

rf_cv_model = rf_cv.fit(train_sample)

# Best RF params
best_rf = rf_cv_model.bestModel.stages[-1]
print(f"Best RF numTrees: {best_rf.getNumTrees}")
print(f"Best RF maxDepth: {best_rf.getMaxDepth()}")
print(f"Best RF minInstancesPerNode: {best_rf.getMinInstancesPerNode()}")

# GBT hyperparameter tuning (3-fold CV):
print("\n=== GBT Cross-Validation ===")

gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=17976)

gbt_param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter,  [10, 20, 30]) \
    .addGrid(gbt.maxDepth, [3, 5, 10]) \
    .addGrid(gbt.stepSize, [0.05, 0.1, 0.2]) \
    .build()

gbt_pipeline = Pipeline(stages=[assembler, gbt])

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=17976,
    parallelism=3
)

gbt_cv_model = gbt_cv.fit(train_sample)

# Best GBT params
best_gbt = gbt_cv_model.bestModel.stages[-1]
print(f"Best GBT maxIter: {best_gbt.getMaxIter()}")
print(f"Best GBT maxDepth: {best_gbt.getMaxDepth()}")
print(f"Best GBT stepSize: {best_gbt.getStepSize()}")

# Compare RF vs GBT on sample train/test:
print("\n=== Performance on 2% Sample ===")

# RF
rf_train_acc = evaluator.evaluate(rf_cv_model.transform(train_sample))
rf_test_acc  = evaluator.evaluate(rf_cv_model.transform(test_sample))
print(f"RF  - Train Accuracy: {rf_train_acc:.4f}  |  Test Accuracy: {rf_test_acc:.4f}")

# GBT
gbt_train_acc = evaluator.evaluate(gbt_cv_model.transform(train_sample))
gbt_test_acc  = evaluator.evaluate(gbt_cv_model.transform(test_sample))
print(f"GBT - Train Accuracy: {gbt_train_acc:.4f}  |  Test Accuracy: {gbt_test_acc:.4f}")

# Part B
print("\n=== Part B: Full Dataset ===")


train_full, test_full = df.randomSplit([0.8, 0.2], seed=17976)
train_full.cache()

train_full_size = train_full.count()
test_full_size  = test_full.count()
print(f"Full train size: {train_full_size}")
print(f"Full test size: {test_full_size}")

# RF pipeline with best params
rf_best = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    seed=17976,
    numTrees=best_rf.getNumTrees,
    maxDepth=best_rf.getMaxDepth(),
    minInstancesPerNode=best_rf.getMinInstancesPerNode()
)
rf_full_pipeline = Pipeline(stages=[assembler, rf_best])
rf_full_model    = rf_full_pipeline.fit(train_full)

rf_full_train_acc = evaluator.evaluate(rf_full_model.transform(train_full))
rf_full_test_acc  = evaluator.evaluate(rf_full_model.transform(test_full))
print(f"RF  Full - Train Accuracy: {rf_full_train_acc:.4f}  |  Test Accuracy: {rf_full_test_acc:.4f}")

# GBT pipeline with best params
gbt_best = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    seed=17976,
    maxIter=best_gbt.getMaxIter(),
    maxDepth=best_gbt.getMaxDepth(),
    stepSize=best_gbt.getStepSize()
)
gbt_full_pipeline = Pipeline(stages=[assembler, gbt_best])
gbt_full_model    = gbt_full_pipeline.fit(train_full)

gbt_full_train_acc = evaluator.evaluate(gbt_full_model.transform(train_full))
gbt_full_test_acc  = evaluator.evaluate(gbt_full_model.transform(test_full))
print(f"GBT Full - Train Accuracy: {gbt_full_train_acc:.4f}  |  Test Accuracy: {gbt_full_test_acc:.4f}")

spark.stop()
