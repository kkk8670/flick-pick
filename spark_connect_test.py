# for local
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Test").getOrCreate()
print(spark.version)

# for google colab
from google.colab import drive
drive.mount('/content/drive')
!ls -1 /content/drive/MyDrive/5003-BigData/data

# for databrick
file_path = "/FileStore/tables/test"
scrpit_path = "/Workspace/Shared/flick-pick"
dbutils.fs.ls(file_path)
df = spark.read.csv(file_path, header=True)
df.show()
# %run /Workspace/Shared/flick-pick/ALS_test