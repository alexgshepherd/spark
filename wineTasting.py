# wineTasting.py
#import findspark
#findspark.init()

import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc= SparkContext()
sqlContext = SQLContext(sc)

house_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true'
	, delimiter=';').load(sys.argv[1])
print(house_df.take(1))

house_df.cache()
house_df.printSchema()

print(house_df.describe().toPandas().transpose())

import six
for i in house_df.columns:
    if not( isinstance(house_df.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to quality for ", i, house_df.stat.corr('""""quality"""""',i))

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['"""""fixed acidity""""', '""""volatile acidity""""'
	, '""""citric acid""""', '""""residual sugar""""', '""""chlorides""""', '""""free sulfur dioxide""""'
	, '""""total sulfur dioxide""""', '""""density""""', '""""pH""""', '""""sulphates""""', '""""alcohol""""']
	, outputCol = 'features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['features', '""""quality"""""'])
vhouse_df.show(3)

train_df = vhouse_df

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='""""quality"""""', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

#lr_model.save('s3://pa2/model')
lr_model.write().overwrite().save('s3://pa2/model')
