import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
# Plot our model prediction
def set_prediction_markers(x_train, y_train, tmp_f_wb):

    plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

    plt.scatter(x_train, y_train, marker="x", c="r", label="Actual Values")
