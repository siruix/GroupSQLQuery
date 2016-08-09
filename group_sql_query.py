# Authors: Sirui Xing <sirui.xing@gmail.com>

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroupSQL(object):
    """ Group SQL query by content similarity

    Parameters
    ----------
    queries : list[string]
        The list type can be Pandas.Series, numpy.array, python list

    n : int, optional, default: 3
        The number of clusters to form as well as the number of
        centroids to generate.

    similarity_metric: string
        similarity metric

"""
    def __init__(self, queries, num_clusters=3, similarity_metric='length'):
        if isinstance(queries, pd.Series):
            self.queries = np.array(queries)

        elif isinstance(queries, list):
            self.queries = np.array(queries)

        elif isinstance(queries, np.ndarray):
            self.queries = queries

        else:
            raise Exception("Unknown queries type")

        self.n = num_clusters
        self.similarity_metric = similarity_metric

    def query_length_metric(self):
        logger.info('Using length metric. ')
        return [len(query) for query in self.queries]

    def words_freq_metric(self):
        """
        Vectorization.
        Turn a collection of text documents into numerical feature vectors.
        """
        logger.info('Using words count metric. ')
        vectorizer = CountVectorizer(min_df=0.01)
        return vectorizer.fit_transform(self.queries)

    def tf_idf(self):
        """
        Only consider alphabetical words. Ignore numbers
        """
        logger.info('Using tf_idf metric. ')
        vectorizer = TfidfVectorizer(min_df=0.01, token_pattern=r"(?u)\b[A-z]+\b")
        results = vectorizer.fit_transform(self.queries)
        logger.info('The learned idf vector: %s'%vectorizer.idf_)
        logger.info('Terms that were ignored: %s'%', '.join(vectorizer.stop_words_))
        return results

    def group(self):
        """Grouping by given similarity metric.

        Returns
        -------
        (label, grouped_id) : (list, dict(np.array) )
        """
        X = None
        if self.similarity_metric == 'length':
            X = self.query_length_metric()
            X = np.array(X, ndmin=2).T

        elif self.similarity_metric == 'WordsFreq':
            X = self.words_freq_metric()

        elif self.similarity_metric == 'tf_idf':
            X = self.tf_idf()

        else:
            raise Exception("Unknown Similarity Metric")

        # l2 normalization has done by TfidfVectorizer
        # normalizer = Normalizer(copy=False)
        # X = normalizer.fit_transform(X)

        logger.info('Using KMeans. ')
        # hardcoded minimize euclidean distance
        km = KMeans(n_clusters=self.n)
        cluster_assigned = km.fit_predict(X)

        grouped_id = {}
        grouped_queries = {}
        for i in range(self.n):
            index = np.where(i == cluster_assigned)[0]
            grouped_id[i] = index
            grouped_queries[i] = self.queries[index]

        logger.info('grouped_queries: %s'%str(grouped_queries))
        return (km.labels_, grouped_id)
