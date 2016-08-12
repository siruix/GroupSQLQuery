"""
    Test of group_sql_query.py
"""
from group_sql_query import GroupSQL
import numpy as np
if __name__ == '__main__':
    with open('queries.txt') as f:
        queries_in = f.readlines()

    queries_in = np.array(queries_in)
    groupSQL = GroupSQL(queries_in, num_clusters=3, similarity_metric='tf_idf')
    label, grouped_id = groupSQL.group()
    print(label)
    print(grouped_id)