import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pandas.read_csv("turbin_data.csv", sep=";", header=0)
data["TIME"] = pandas.to_datetime(data["TIME"])
data = data[(0 <= data["SIMULATED_EFF"]) & (data["SIMULATED_EFF"] <= 100)]
data = data[data["DISCHARGE_PRES"] > 6]

X = data[["AIR_IN_TEMP", "AIR_IN_PRES", "DISCHARGE_TEMP", "DISCHARGE_PRES"]]
y = data["SIMULATED_EFF"]

reg_model = linear_model.LinearRegression()
reg_model.fit(X, y)

pred_y = reg_model.predict(X)

plt.plot(data["TIME"], data["SIMULATED_EFF"], "ro", data["TIME"], pred_y, "bo")
plt.show()
