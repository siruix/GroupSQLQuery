from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroupSQL_Spark(object):
    """ Group SQL query by content similarity

    Parameters
    ----------
    queries : A Spark DataFrame with Column 'query_body'

    n : int, optional, default: 20
        The number of clusters to form as well as the number of
        centroids to generate.

"""
    def __init__(self, df, queryColName='query_body', numFeatures=(1 << 10), numClusters=20, minDocFreq=3):
        self.df = df
        assert(isinstance(queryColName, str))
        self.query_colname = queryColName
        assert(isinstance(numFeatures, int))
        self.num_features = numFeatures
        assert(isinstance(numClusters, int))
        self.n = numClusters
        assert(isinstance(minDocFreq, int))
        self.min_doc_freq = minDocFreq

    def group(self):
        reTokenizer = RegexTokenizer(inputCol=self.query_colname, outputCol="words", minTokenLength=2) #, pattern='\W'
        hashingTF = HashingTF(numFeatures=self.num_features, inputCol="words", outputCol="tf")
        idf = IDF(minDocFreq=self.min_doc_freq, inputCol="tf", outputCol="idf")
        kmeans = KMeans(featuresCol="idf", predictionCol="cluster_id", k=self.n)

        pipeline = Pipeline(stages=[reTokenizer, hashingTF, idf, kmeans])
        model = pipeline.fit(self.df)
        prediction = model.transform(self.df)
        return prediction

