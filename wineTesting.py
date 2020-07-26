#need to load in testing dataset

import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc= SparkContext()
sqlContext = SQLContext(sc)

test_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true'
	, delimiter=';').load(sys.argv[1])
print(test_df.take(1))

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['"""""fixed acidity""""', '""""volatile acidity""""'
	, '""""citric acid""""', '""""residual sugar""""', '""""chlorides""""', '""""free sulfur dioxide""""'
	, '""""total sulfur dioxide""""', '""""density""""', '""""pH""""', '""""sulphates""""', '""""alcohol""""']
	, outputCol = 'features')
vtest_df = vectorAssembler.transform(test_df)
vtest_df = vtest_df.select(['features', '""""quality"""""'])
vtest_df.show(3)

from pyspark.ml.regression import LinearRegressionModel
lr_model = LinearRegressionModel.load('s3://pa2/model')
lr_predictions = lr_model.transform(vtest_df)
lr_predictions.select('prediction','""""quality"""""','features').show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='""""quality"""""',metricName='r2')
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
