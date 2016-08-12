# Authors: Sirui Xing <sirui.xing@gmail.com>

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    with open('queries.txt') as f:
        queries_in = f.readlines()

    queries_in = np.array(queries_in)

    vectorizer = TfidfVectorizer(min_df=0.01, token_pattern=r"(?u)\b[A-z]+\b", use_idf=False)
    X = vectorizer.fit_transform(queries_in)

    km = KMeans(n_clusters=3)
    label = km.fit_predict(X)
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.toarray())

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=label, cmap=plt.cm.Set1)

    plt.show()
