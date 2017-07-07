import pandas, numpy
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pandas.read_csv("turbin_data.csv", sep=";", header=0)
data["TIME"] = pandas.to_datetime(data["TIME"])

data = data.dropna(axis=0, how="any")
data = data[data['DISCHARGE_PRES'] > 6]
data = data[(0.8 <= data['AIR_IN_PRES']) & (data['AIR_IN_PRES'] <= 1.1)]
data = data.reset_index()

split_time = data["TIME"][849]#int(len(data)*0.58)]
data = data.set_index("TIME")
print " >> After preprocessing %d data points remaining" % len(data)

poly_data = lambda X : PolynomialFeatures(degree=2).fit_transform(X)
X = data[["AIR_IN_TEMP", "AIR_IN_PRES", "DISCHARGE_TEMP", "DISCHARGE_PRES"]]
y = data["SIMULATED_EFF"]

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=4)
#X_train, X_test = X[:split_time:], X[split_time::]
#y_train, y_test = y[:split_time:], y[split_time::]

reg = linear_model.LinearRegression()
reg = reg.fit(poly_data(X_train), y_train)
# print "Trained poly: coef: %s, constant: %f" % (str(reg.coef_), reg.intercept_)

print "Score on training set %s" % reg.score(poly_data(X_train), y_train)
print "Score on test set %s" % reg.score(poly_data(X_test), y_test)

plt.plot(
        y.index, y, "ro",
        X_train.index, reg.predict(poly_data(X_train)), "bo",
        X_test.index, reg.predict(poly_data(X_test)), "go"
        )
plt.show()
