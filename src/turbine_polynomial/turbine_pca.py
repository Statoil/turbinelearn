import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import turbine_regression

def pca(data_file):
    data = turbine_regression.load_data(data_file)
    data = turbine_regression.preprocess_data(data)
    X = data[turbine_regression.FEATURES + [turbine_regression.TARGET]]
    y = data[turbine_regression.TARGET]
    pca = PCA(n_components=2)
    X_2D = pca.fit(X).transform(X)
    explained = pca.explained_variance_ratio_
    print('Explained variance ratio: %.5f, %.5f' % (explained[0], explained[1]))

    X_1 = [x[0] for x in X_2D]
    X_2 = [x[1] for x in X_2D]

    ### Plot the 2 dimensions with y as color of circle
    cmap = sns.cubehelix_palette(as_cmap=True)

    f, ax = plt.subplots()
    points = ax.scatter(X_1, X_2, c=y, cmap=cmap)
    f.colorbar(points)
    plt.show()


if __name__ == '__main__':

    data_files = ["data/LOCO_B_HGA.csv",
                  "data/LOCO_B_HGB.csv",
                  "data/LOCO_B_HTA.csv",
                  "data/LOCO_B_HTB.csv",
                  "data/LOCO_C_HGA.csv",
                  "data/LOCO_C_HGB.csv",
                  "data/LOCO_C_HTA.csv",
                  "data/LOCO_C_HTB.csv"]
    for f in data_files:
        pca(f)
