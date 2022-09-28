import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv('data_2c_prev.csv')
    data[['Height', 'Weight', 'Age']] = data[['Height', 'Weight', 'Age']].apply(pd.to_numeric)
    X = data[['Height', 'Weight', 'Age']]
    Y = data[['Gender']]
    return X, Y


def load_test_data():
    data = pd.read_csv('data_2c_prev.csv')
    return data


def load_test_data_age():
    data = pd.read_csv('data_2c_prev.csv')
    data_temp = data.copy()
    data_temp = data_temp.drop(['Age'], axis=1)
    return data_temp


def bulding_feature_matrix(X):
    X.insert(0, "P0", 1)

    return


def plot_3D(X, Y, theta):
    theta = theta.flatten()
    # create x,y
    xx, yy = np.meshgrid(X[:, 1], X[:, 2])

    theta_0 = theta[0]
    theta_1 = theta[1]
    theta_2 = theta[2]
    theta_3 = theta[3]

    zz = -(theta_0 + theta_1 * xx + theta_2 * yy) * 1. / theta_3

    # plot the surface
    plt3d = plt.figure(figsize=(16, 12), dpi=80).gca(projection='3d')

    # Scatter target ouput
    i = 0
    for x in X:
        x1 = x[1]
        y1 = x[2]
        z1 = x[3]
        target = Y[i]
        if (target == 1):
            m = plt3d.scatter(x1, y1, z1, marker='o', color='green', s=20)
        else:
            w = plt3d.scatter(x1, y1, z1, marker='o', color='red', s=20)
        i = i + 1

    plt.legend((m, w), ('Man', 'Woman'), scatterpoints=1, loc='upper left', fontsize=8)
    # Plot function
    plt3d.plot_surface(xx, yy, zz, color='blue', alpha=0.05)
    plt3d.set_xlabel('Height')
    plt3d.set_ylabel('Weight')
    plt3d.set_zlabel('Age')
    plt.title("Logistically Regressed Hyperplane", fontsize=20, color='blue', loc='left')
    plt.show()


def sigmoid(z):
    z = z.flatten()
    sig = 1.0 / (1 + np.exp(-z))
    return sig


def logistic_regression(X, Y):
    # Step size
    alpha = 0.001
    # No. of iterations
    iterations = 10000
    # number of features = n, number of data points = m
    m, n = X.shape

    # Pick some random values to start with
    # theta = np.random.rand(1,n)
    theta = np.zeros((1, n))
    # print('theta;', theta)
    Y = Y.reshape((m, 1))

    # gradient acent
    for i in range(iterations):
        linear_function = sigmoid(np.dot(X, theta.T))
        linear_function = linear_function.reshape((m, 1))
        diff = Y - linear_function
        theta1 = alpha * (1 / m) * np.dot(diff.T, X)
        theta = theta + (alpha * np.dot(diff.T, X))

    return theta


def prediction(x_test, theta):
    m = len(x_test)
    ones = np.ones((m, 1))
    x_test = np.column_stack((ones, x_test))
    z = np.dot(x_test, theta.T)
    y = (sigmoid(z) >= 0.5).astype(int)

    return y


def leave_one_out_evaluation(theta, age):
    pre_data = load_test_data() if age == 1 else load_test_data_age()
    # pre_data = pd.Series(np.where(pre_data.Gender.values == 'M', 1, 0),pre_data.index)
    loop_var = len(pre_data.index)
    error_count = 0
    if age == 0:
        theta = theta.flatten()
        theta = np.delete(theta, 3, 0)
        theta = theta.reshape((1, 3))

    for i in range(loop_var):

        dropped_row = pre_data.iloc[[i], :]
        data = pre_data.drop(pre_data.index[i])
        test_data = dropped_row[['Height', 'Weight', 'Age']].to_numpy() if age == 1 else dropped_row[
            ['Height', 'Weight']].to_numpy()

        expect = dropped_row[['Gender']].to_numpy()
        expected_prediction = np.array(expect[:, -1])
        result = prediction(test_data, theta)
        gen = 'M' if result == 1 else 'W'
        if gen != expected_prediction:
            error_count = error_count + 1
    percent_of_error = error_count / loop_var * 100

    return percent_of_error


def main():
    X, Y = load_data()

    Y = pd.Series(np.where(Y.Gender.values == 'M', 1, 0), Y.index)

    bulding_feature_matrix(X)

    X = X.to_numpy()
    Y = Y.to_numpy()

    parameters = logistic_regression(X, Y)

    print('\n\n')
    print('\033[1m''Printing the parameters obtain from Logical Regression:' + '\033[0m')
    for i in range(len(parameters[0, :])):
        print("Parameter", i, ':', parameters[0, i])

    # Ploting
    print('\n\n')
    print('\033[1m' + 'Printing the function obtain from Logical Regression:' + '\033[0m')
    plot_3D(X, Y, parameters)

    # Leave one out evaluation
    percent_of_error = leave_one_out_evaluation(parameters, 1)
    print('\n\n')
    print('\033[1m' + 'Printing the result of leave one out evaluation:' + '\033[0m')
    print('\033[1m' + 'Error percentage:', percent_of_error, '\033[0m')

    # Leave one out evaluation
    percent_of_error = leave_one_out_evaluation(parameters, 0)
    print('\n\n')
    print('\033[1m' + 'Printing the result of leave one out evaluation when age column is dropped:' + '\033[0m')
    print('\033[1m' + 'Error percentage:', percent_of_error, '\033[0m')


if __name__ == '__main__':
    main()
