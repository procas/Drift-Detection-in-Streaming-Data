# # Model training & testing steps:
# 
# - Data preprocessing
# - Cleaning
# - Encoding categorical variable - family
# - Feature Selection -> RFE
# - K Means Clustering with 2 classes
# - Autoencoder sample
# 
# Cleaning: 
# 
# - dropna()
# - deal with missing values with avg
# Preprocessing:
# 
# - Feature extraction with RFE
# - New dataset with important features
# - Encoding categorical variables like service and protocol type in the reduced dataset: using Onehotencoder


import warnings
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import plotly.express as px
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


# load data
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(url, names=names)
dataframe = pd.read_csv(r'C:\Users\Proma_Mukherjee\Downloads\KDDdataset.csv')
array = dataframe.values
X = array[:,5:41]
Y = array[:,41]
# # feature extraction
# pca = PCA(n_components=30)
# fit = pca.fit(X)
# # summarize components
# print("Explained Variance: %s" % fit.explained_variance_ratio_)
# print(fit.components_)
# #fit.ranking_

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 1)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


#2 13 16 22 23 24 29 30 31 32
#service root_shell num_file_creations count srv_count serror_rate diff_srv_rate
#srv_diff_host_rate dst_host_count dst_host_srv_count
df.to_excel('KDD.xlsx')
df

#df = dataframe[['service root_shell','num_file_creations','count srv_count','serror_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count']]
# df = dataframe[['service','root_shell','num_file_creations','count','srv_count','serror_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','class']]
# df


le = LabelEncoder()
df['service']= le.fit_transform(df['service'])
# data['Geography']= le.fit_transform(data['Geography'])


# for i in range(0,len(dataframe.columns)):
#     print(i," ",dataframe.columns[i])


X = df[:1000].values[:,0:10]
y = df[:1000].values[:,10]

# # Auto encoder check


 
# SCALE EACH FEATURE INTO [0, 1] RANGE
sX = minmax_scale(X, axis = 0)
ncol = sX.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(sX, y, train_size = 0.5, random_state = seed(2017))
 
### AN EXAMPLE OF SIMPLE AUTOENCODER ###
# InputLayer (None, 10)
#      Dense (None, 5)
#      Dense (None, 10)
 
input_dim = Input(shape = (ncol, ))
# DEFINE THE DIMENSION OF ENCODER ASSUMED 3
encoding_dim = 10
# DEFINE THE ENCODER LAYER
encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
# DEFINE THE DECODER LAYER
decoded = Dense(ncol, activation = 'sigmoid')(encoded)
# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(input_dim, decoded)
# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.fit(X_train, X_train, batch_size = 100, shuffle = True, validation_data = (X_test, X_test))
# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input_dim, encoded)
encoded_input = Input(shape = (encoding_dim, ))
encoded_out = encoder.predict(X_test)
encoded_out[0:10]


# # KMeans Check

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])

wcss = []
dataframe = pd.read_csv(r'C:\Users\Proma_Mukherjee\Downloads\KDDdataset.csv')

array = df[:100].values
#X = array[:,35:41]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=100, n_init=100, random_state=42)
pred_y = kmeans.fit_predict(X)
# plt.scatter(X[:,1], X[:,9], color='blue')
# plt.scatter(X[:,2], X[:,9], color='blue')
# plt.scatter(X[:,3], X[:,9], color='blue')
# plt.scatter(X[:,4], X[:,9], color='blue')
# plt.scatter(X[:,5], X[:,9], color='blue')
# plt.scatter(X[:,6], X[:,9], color='blue')
plt.scatter(X[:,8], X[:,9], color='green')
plt.scatter(X[:,3], X[:,9], color='blue')
plt.scatter(kmeans.cluster_centers_[:, 3],kmeans.cluster_centers_[:, 9], s=1000, c='red')
plt.scatter(kmeans.cluster_centers_[:, 8],kmeans.cluster_centers_[:, 9], s=1000, c='yellow')
plt.show()


fig2 = px.scatter_3d(df, x="class", y="service",z="count",
                     color="diff_srv_rate",size="count")
fig2.update_layout(title="5 Features Representation")
fig2.show()

