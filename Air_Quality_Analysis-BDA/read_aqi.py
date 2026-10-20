
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("AQI Category Analysis") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs://localhost:9000/aqi/cleaned",
    header=True,
    inferSchema=True
)

category_count = df.groupBy("AQI_Bucket") \
    .count() \
    .orderBy("count", ascending=False)

print("AQI Category Count:")
category_count.show(truncate=False)

category_count.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("hdfs://localhost:9000/aqi/category_analysis")

print("Category analysis saved to HDFS.")

spark.stop()
