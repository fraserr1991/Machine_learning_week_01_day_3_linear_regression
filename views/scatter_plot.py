import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

def create_scatter_plot(x_train, y_train):

    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title("Housing Prices")
    plt.ylabel("Price (in 1000s of dollars)")
    plt.xlabel("Size (1000 sqft)")
    plt.show()
