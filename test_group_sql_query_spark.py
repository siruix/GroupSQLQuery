"""
    Test of group_sql_query_spark.py
"""
from group_sql_query_spark import GroupSQL_Spark
import numpy as np
if __name__ == '__main__':
    with open('queries.txt') as f:
        queries_in = f.readlines()

    queries_in = np.array(queries_in)
    groupSQL = GroupSQL_Spark(queries_in, queryColName='query_body', numFeatures=(1 << 10), numClusters=num_distinct_queries, minDocFreq=3)
    prediction_df = groupSQL.group()
    print(prediction_df)
