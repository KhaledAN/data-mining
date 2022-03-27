import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import seaborn as sns

cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

# call and process data
def get_data():
    # call data file
    url ='./input/dataset.csv'
    df = pd.read_csv(url,index_col=False, header=None, names=cols)

    # replace '?' with 'np.nan'
    df = df.replace({'?':np.nan})

    # drop rows with nans for now, only looses 6 rows.
    df.dropna(inplace=True)
    df = df.astype('float64')

    # to change the target data to 0 and 1
    # 0 means 'No heart disease', 1 means 'heart disease'
    df['num'] = df['num']>0
    df['num'] = df['num'].map({False:0, True:1})
    
    return df

def hierarchical():
    df = get_data()
    y = df.pop('num').values
    #x=df.iloc[:,0:11]
  
    x = pd.DataFrame(scale(df.values),columns=cols[0:13])
    pca = PCA(n_components=2,svd_solver='full')
    X_pca = pca.fit_transform(x)
    X_reduced, X_test_reduced, Y_Train, Y_Test = train_test_split(X_pca, y, test_size = 0.30, random_state = 101)
    dendrogram = sch.dendrogram(sch.linkage(X_reduced, method='ward'))
    plt.savefig('result/hierarchical-dendrogram.png', dpi=520, format='png', bbox_inches='tight')
    hc = AgglomerativeClustering(n_clusters=2, affinity = 'euclidean', linkage = 'ward')
    hierarchicalpredictions = hc.fit_predict(X_test_reduced)
    print('hierarchical: ', accuracy_score(Y_Test,hierarchicalpredictions))


def kMean():
    df = get_data()
    y = df.pop('num').values
    x = pd.DataFrame(scale(df.values),columns=cols[0:13])
    estimator = KMeans(n_clusters=2, init='k-means++', random_state=42)
    estimator.fit_predict(x)
    draw_kMean_clusters(x,estimator)
    print('kMean: ', accuracy_score(y,estimator.predict(x)))


def draw_kMean_clusters(x,estimator):
    # run PCA to reduce the dimension 
    table = x.copy()
    table['cluster'] = estimator.labels_
    
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(table)
    pca_df = pd.DataFrame(new_data,columns=['principal component1','principal component2'])

    # drawing
    sns.scatterplot(x="principal component1", y="principal component2", hue=table['cluster'], data=pca_df)
    plt.title('K-means Clustering in 2D')
    # Edit 'path_to_save' as your directory to save the result diagram 
    plt.savefig('result/kmeans_result.png')
    plt.show()


kMean()
hierarchical()