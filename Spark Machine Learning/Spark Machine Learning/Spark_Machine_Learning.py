from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
import sys
import gc

spark = SparkSession.builder.appName('ML_project').getOrCreate()

#This function is used to load big json file
def read_json(filePath):
    df = spark.read.json(filePath)
    return df

def main():
    #here set your data location path
    dataPath = 'Musical_Instruments_5.json'# this dataset is from http://jmcauley.ucsd.edu/data/amazon/links.html
    #call function to read json data format
    df = read_json(dataPath)
    #df.printSchema()
    indexer1 = StringIndexer(inputCol='asin',outputCol='product_id')
    indexed = indexer1.fit(df).transform(df)
    indexer2 = StringIndexer(inputCol='reviewerID',outputCol='reviewerID_index')
    indexed = indexer2.fit(indexed).transform(indexed)
    #print(indexed.head(1))
    assmbler = VectorAssembler(inputCols=[
                                          'product_id',
                                          'reviewerID_index'],
                               outputCol='features')
    ouput = assmbler.transform(indexed)
    #free memory from unused memory allocation
    colloter = gc.collect()
    print('free {} memory items ...'.format(colloter))
    #split data randomly for 70% train and 30% for test
    train_data, test_data = ouput.select('features','overall').randomSplit([0.7,0.3])
    #
    #train_data.describe().show()
    #test_data.describe().show()
    #use Linear Regression and set the label attribute
    lr = LinearRegression(labelCol ='overall')
    #run classifer train 
    lr_model = lr.fit(train_data)
    #evaluate the model with test data
    resultes = lr_model.evaluate(test_data)
    #print square error and r2
    print('Square error = {}\nR2 = {}'.format(resultes.rootMeanSquaredError,resultes.r2))
    #stop spark session
    spark.stop()
if __name__ =='__main__':
    main() 