import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster


df = pd.read_csv(r'C:\Users\ANANTH ANAND P B\Downloads\archive (3)\Mall_Customers.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.info())

new_df = df.iloc[:, -2:]
print(new_df.head())

error = []
for i in range(1,11):
    kmeans = cluster.KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(new_df)
    error.append(kmeans.inertia_)

plt.plot(range(1, 11), error)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('Error')
plt.show()

km_model = cluster.KMeans(n_clusters=5, init='k-means++', random_state=0)
km_model.fit(new_df)

pred = km_model.predict(new_df)
print(pred)

print(km_model.predict([[66, 98]]))
print(km_model.predict([[30, 59]]))
print(km_model.predict([[40, 50]]))
print(km_model.predict([[59, 99]]))