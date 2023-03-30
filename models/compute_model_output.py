import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

def compute_model_output(x, w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb