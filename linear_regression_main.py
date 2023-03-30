import numpy as np
import matplotlib.pyplot as plt

from models.compute_model_output import compute_model_output
from views.scatter_plot import create_scatter_plot
from views.prediction_markers import set_prediction_markers

plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

#.shape is getting the length of the training data, returning a tuple
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

create_scatter_plot(x_train, y_train)

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")
print(f"w: {w}")
print(f"b: {b}")

tmp_f_wb = compute_model_output(x_train, w, b,)

set_prediction_markers(x_train, y_train, tmp_f_wb)
create_scatter_plot(x_train, y_train)
