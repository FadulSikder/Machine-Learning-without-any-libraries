import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def plot(X, Y, x_func, y_funcs, preds, tau):
    # theta = theta.flatten()
    X = X.flatten()
    Y = Y.flatten()

    # x = np.linspace(-np.pi*3, np.pi*3, 541)
    # y = theta[0] + theta[1]*x
    # for i in range(1,depths+1):
    #     temp = theta[2*i] * sin_func(x,i) + theta[2*i+1]*cos_func(x,i)
    #     y = y + temp

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
    plt.plot(x_func, y_funcs, 'b', linewidth=3)
    plt.scatter(X, preds, c='red')
    # Plot the points
    plt.scatter(X, Y, c='green')

    # giving a title to my graph
    if len(X) > 21:
        title = 'Locally Weighted Regression Function \nat bandwidth ' + str(tau) + ' with all 121 elements'
    else:
        title = 'Locally Weighted Regression Function \nat bandwidth ' + str(tau) + ' with only 20 elements'
    plt.title(title, fontsize=20, color='blue', loc='left')
    # function to show the plot
    plt.show()


def bulding_feature_matrix(X):
    X.insert(0, "P0", 1)
    return


def compute_error(Y_test, y_test_preds):
    Y_test = Y_test.flatten()
    rmse = np.sqrt(np.square(y_test_preds - Y_test).mean())

    return rmse


def getW(query_point, X, tau):
    m = X.shape[0]
    W = np.mat(np.eye(m))
    for i in range(m):
        xi = X[i]
        W[i, i] = np.exp(np.dot((xi - query_point), (xi - query_point).T) / (-2 * tau * tau))
    return W


def locally_weighted_rgression(X, Y, query_x, tau):
    qx = np.mat([1, query_x])

    W = getW(qx, X, tau)

    theta = np.linalg.pinv(X.T * (W * X)) * (X.T * (W * Y))
    pred = np.dot(qx, theta)

    return pred


def procedures(X, Y, X_test, Y_test):
    # Dataframe conversion  and manupulation
    X_plot = np.array(X)
    X_loop = X_plot.flatten()
    Y_temp = Y.to_numpy()
    Y_test = Y_test.to_numpy()
    rmse = np.zeros(3)
    X_temp = X.copy()
    X_test = X_test.to_numpy()
    X_test = np.array(X_test).flatten()
    bulding_feature_matrix(X_temp)
    X_temp = X_temp.to_numpy()
    i = 0

    tau_value = [0.2, 0.1, 0.05]

    print('\033[1m' + 'Result of Experiment 2(b):' + '\033[0m') if len(X.index) > 21 else print(
        '\033[1m' + 'Result of Experiment 2(b) only with dataset of 20 elements(Question 1(d)):' + '\033[0m')
    for tau in tau_value:
        # Variable Declaration
        preds = []
        y_funcs = []
        y_test_preds = []

        # Computing the function for each data point
        for query_x in X_loop:
            pred = locally_weighted_rgression(X_temp, Y_temp, query_x, tau)
            preds.append(pred)
        preds = np.array(preds)
        preds2 = preds.flatten()

        x_func = np.linspace(-np.pi, np.pi, 181)
        for query_x in x_func:
            y_func = locally_weighted_rgression(X_temp, Y_temp, query_x, tau)
            y_funcs.append(y_func)
        y_funcs = np.array(y_funcs)
        y_funcs2 = y_funcs.flatten()

        # Ploting
        print('\n\n')
        print('\033[1m' + 'Printing the function obtain from Locally Weighted Linear Regression with bandwidth:', tau,
              '\033[0m')
        print('\n\n')
        plot(X_plot, Y_temp, x_func, y_funcs2, preds2, tau)

        # Error calculation on Test data
        for query_x in X_test:
            y_test_pred = locally_weighted_rgression(X_temp, Y_temp, query_x, tau)
            y_test_preds.append(y_test_pred)
        y_test_preds = np.array(y_test_preds)
        y_test_preds = y_test_preds.flatten()
        rmse[i] = compute_error(Y_test, y_test_preds)
        i = i + 1

    print('\n\n')
    print('\033[1m' + 'tau = 0.2 was given in my data set. However, I experiment with two other tau value.' + '\033[0m')
    print('\033[1m' + 'Result of Experiment 2(c):' + '\033[0m') if len(X.index) > 21 else print(
        '\033[1m' + 'Result of Experiment 2(c) only with dataset of 20 elements(Question 1(d)):' + '\033[0m')
    for j in range(0, 3):
        print('Root Mean Square Error at bandwidth ', tau_value[j], ':', rmse[j])
    print("Funtion with mimimum RMSE is at bandwidth: ", tau_value[np.argmin(rmse)])


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
