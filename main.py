import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import pandas
import matplotlib.pyplot as plt

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']


def get_data():
    url = './input/dataset.csv'
    csvFile = pandas.read_csv(url, index_col=False, header=None, names=cols)
    csvFile = csvFile.replace({'?': np.nan})
    csvFile.dropna(inplace=True)
    csvFile = csvFile.astype('float64')
    csvFile['num'] = csvFile['num'] > 0
    csvFile['num'] = csvFile['num'].map({False: 0, True: 1})
    return csvFile


def hierarchical():
    dataFile = get_data()
    y = dataFile.pop('num').values
    x = pandas.DataFrame(scale(dataFile.values), columns=cols[0:13])
    pca = PCA(n_components=2, svd_solver='full')
    X_pca = pca.fit_transform(x)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        X_pca, y, test_size=0.30, random_state=101)
    drawHierarchicalDendrogram(X_Train)
    hc = AgglomerativeClustering(
        n_clusters=2, affinity='euclidean', linkage='ward')
    hierarchicalpredictions = hc.fit_predict(X_Test)
    accuracy = accuracy_score(Y_Test, hierarchicalpredictions)
    print('hierarchical: ', accuracy)


def drawHierarchicalDendrogram(X_Train):
    sch.dendrogram(sch.linkage(X_Train, method='ward'))
    plt.title('Hierarchical')
    plt.savefig('result/hierarchical-dendrogram.png',
                dpi=520, format='png', bbox_inches='tight')
    plt.show()


def kMean():
    dataFile = get_data()
    Y = dataFile.pop('num').values
    X = pandas.DataFrame(scale(dataFile.values), columns=cols[0:13])
    kmeansRes = KMeans(n_clusters=2, init='k-means++', random_state=42)
    kmeansRes.fit_predict(X)
    draw_kMean_clusters(X, kmeansRes)
    accuracy = accuracy_score(Y, kmeansRes.predict(X))
    print('kMean: ', accuracy)


def draw_kMean_clusters(X, kmeansRes):
    data = X.copy()
    data['cluster'] = kmeansRes.labels_

    pca = PCA(n_components=2)
    new_data = pca.fit_transform(data)
    pca_df = pandas.DataFrame(
        new_data, columns=['X', 'Y'])

    sns.scatterplot(x="X", y="Y",
                    hue=data['cluster'], data=pca_df)
    plt.title('K-means')
    plt.savefig('result/kmeans_result.png')
    plt.show()


kMean()
hierarchical()
