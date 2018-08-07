# GroupSQLQuery
An utility to group relevant SQL queries by similarity metrics

This project is to group similar SQL Query together. 
The input is a file containing lots of SQL queries. Those queries are (1) transformed to similarity vectors, (2) calculate similarity and grouping using K-means. 
It then plot the high dimentional points into 2-D space using t-SNE. 

The code is compatible with Spark ML. 
