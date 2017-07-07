import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pandas.read_csv("turbin_data.csv", sep=";", header=0)

data = data.dropna(axis=0, how="any")
data = data[data['DISCHARGE_PRES'] > 6]
data = data[(0.8 <= data['AIR_IN_PRES']) & (data['AIR_IN_PRES'] <= 1.1)]

X = data[["AIR_IN_TEMP", "AIR_IN_PRES", "DISCHARGE_TEMP", "DISCHARGE_PRES"]]
y = data["SIMULATED_EFF"]

pca = PCA(n_components=2)
X_2D = pca.fit(X).transform(X)
explained = pca.explained_variance_ratio_
print('Explained variance ratio: %g, %g' % (explained[0], explained[1]))

X_1 = [x[0] for x in X_2D]
X_2 = [x[1] for x in X_2D]

### Plot the 2 dimensions with y as color of circle
cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
points = ax.scatter(X_1, X_2, c=y, cmap=cmap)
f.colorbar(points)
plt.show()
