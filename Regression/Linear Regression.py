import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k = 8


def load_data():
    data = pd.read_csv('data_1b.csv')
    data[['X', 'Y']] = data[['X', 'Y']].apply(pd.to_numeric)

    X = data[['X']]
    Y = data[['Y']]

    return X, Y


def test_load_data():
    data = pd.read_csv('data_1c.csv')
    data[['X', 'Y']] = data[['X', 'Y']].apply(pd.to_numeric)

    X = data[['X']]
    Y = data[['Y']]

    return X, Y


def plot(X, Y, theta, depths):
    theta = theta.flatten()
    X = X.flatten()
    Y = Y.flatten()

    x = np.linspace(-np.pi * 3, np.pi * 3, 541)
    y = theta[0] + theta[1] * x
    for i in range(1, depths + 1):
        temp = theta[2 * i] * sin_func(x, i) + theta[2 * i + 1] * cos_func(x, i)
        y = y + temp

    # Define the x and y ranges, and the tick interval for both axes.
    xmin, xmax, ymin, ymax = -3, 3, 0, 6
    ticks_frequency = 1

    # Create a figure and an axes object. Also set the face color.
    # This will cover transparent margins.
    fig, ax = plt.subplots(figsize=(16, 12), dpi=80)
    fig.patch.set_facecolor('#ffffff')

    # Apply the ranges to the axes.
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1))

    # Set both axes to the zero position.
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Hide the top and right spines.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set the x and y labels, and add an origin label.
    ax.set_xlabel('$x$', size=14, labelpad=-24, x=1.02)
    ax.set_ylabel('$y$', size=14, labelpad=-21, y=1.02, rotation=0)

    # Now create the x and the y ticks, and apply them to both axes.
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Finally, add a grid.
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Plot the function
    plt.plot(x, y, 'r')
    # Plot the points
    plt.scatter(X, Y, c='green')

    # giving a title to my graph
    if len(X) > 21:
        title = 'Function at depths ' + str(depths) + ' with all 121 elements'
    else:
        title = 'Function at depths ' + str(depths) + ' with only 20 elements'
    plt.title(title, fontsize=20, color='blue', loc='left')
    # function to show the plot
    plt.show()


def sin_func(X, i):
    return np.sin(X * k * i)


def cos_func(X, i):
    return np.cos(X * k * i)


def bulding_feature_matrix(X, depths):
    X.insert(0, "P0", 1)
    for j in range(1, depths + 1):
        X['P' + str(2 * j)] = X[['X']].apply(sin_func, i=j)
        X['P' + str(2 * j + 1)] = X[['X']].apply(cos_func, i=j)

    return


def regression(X, Y):
    # Step size
    alpha = 0.01
    # No. of iterations
    iterations = 2000
    # number of features = n, number of data points = m
    m, n = X.shape

    # Pick some random values to start with
    theta = np.random.rand(1, n)

    # gradient descent
    for i in range(iterations):
        linear_function = np.dot(X, theta.T)
        error = linear_function - Y
        theta = theta - (alpha * (1 / m) * np.dot(error.T, X))

    return theta


def compute_error(X_test, Y_test, theta):
    predictions = np.dot(X_test, theta.T)
    rmse = np.sqrt(np.square(predictions - Y_test).mean())

    return rmse


def procedures(X, Y, X_test, Y_test):
    # Dataframe conversion  and manupulation
    X_plot = np.array(X)
    Y_temp = Y.to_numpy()
    Y_test = Y_test.to_numpy()
    rmse = np.zeros(7)

    print('\033[1m' + 'Result of Experiment 1(b):' + '\033[0m') if len(X.index) > 21 else print(
        '\033[1m' + 'Result of Experiment 1(b) only with dataset of 20 elements(Question 1(d)):' + '\033[0m')
    for depths in range(0, 7):
        # Dataframe conversion  and manupulation
        X_temp = X.copy()
        X_test_copy = X_test.copy()
        bulding_feature_matrix(X_temp, depths)
        X_temp = X_temp.to_numpy()

        # Linear Regression
        parameters = regression(X_temp, Y_temp)
        print('\n\n')
        print('\033[1m''Printing the parameters obtain from Linear Regression at depths', depths, ':' + '\033[0m')
        for i in range(len(parameters[0, :])):
            print("Parameter", i, ':', parameters[0, i])

        # Ploting
        print('\n\n')
        print('\033[1m' + 'Printing the function obtain from Linear Regression at depths', depths, ':' + '\033[0m')
        plot(X_plot, Y_temp, parameters, depths)

        # Error calculation on Test data
        bulding_feature_matrix(X_test_copy, depths)
        X_test_copy = X_test_copy.to_numpy()
        rmse[depths] = compute_error(X_test_copy, Y_test, parameters)

    print('\n\n')
    print('\033[1m' + 'Result of Experiment 1(c):' + '\033[0m') if len(X.index) > 21 else print(
        '\033[1m' + 'Result of Experiment 1(c) only with dataset of 20 elements(Question 1(d)):' + '\033[0m')
    for depths in range(0, 7):
        print('Root Mean Square Error at depths', depths, ':', rmse[depths])
    print("Funtion with mimimum RMSE is at depths: ", np.argmin(rmse))


def main():
    X, Y = load_data()
    X_20 = X.head(20)
    Y_20 = Y.head(20)
    X_test, Y_test = test_load_data()

    procedures(X, Y, X_test, Y_test)

    print('\n\n')
    print('\033[1m' + 'Result of Experiment 1(b) and 1(c) only with dataset of 20 elements(Question 1(d)):' + '\033[0m')
    print('\n\n')
    procedures(X_20, Y_20, X_test, Y_test)

    return


if __name__ == '__main__':
    main()
