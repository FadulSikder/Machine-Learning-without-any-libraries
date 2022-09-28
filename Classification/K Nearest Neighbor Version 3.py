import numpy as np
import pandas as pd


def load_data():
    data = pd.read_csv('data_2c.csv')
    data = data.drop(['Age'], axis=1)
    data[['Height', 'Weight']] = data[['Height', 'Weight']].apply(pd.to_numeric)
    return data


def cartesian_distance(feature, converted_array_of_data_point):
    distance_matrix = np.power((feature - converted_array_of_data_point), 2)

    return np.sqrt(np.sum(distance_matrix, axis=1))


def knn_algorithm(feature, labels, data_point, k):
    row = feature.shape[0]
    collum = feature.shape[1]

    converted_array_of_data_point = np.full((row, collum), data_point)

    distance = cartesian_distance(feature, converted_array_of_data_point)

    distance_labels = np.column_stack((distance, labels))

    distance_labels = distance_labels[np.argsort(distance_labels[:, 0])]

    k_nearest_points = distance_labels[:k, :]

    k_nearest_labels = np.array(k_nearest_points[:, -1])

    men_occurencies = np.count_nonzero(k_nearest_labels == 'M')
    women_occurencies = np.count_nonzero(k_nearest_labels == 'W')

    result = str("M") if men_occurencies > women_occurencies else str("W")

    return result


def leave_one_out_evaluation(k):
    pre_data = load_data()

    loop_var = len(pre_data.index)
    error_count = 0

    for i in range(loop_var):
        dropped_row = pre_data.iloc[[i], :]
        data = pre_data.drop(pre_data.index[i])
        feature = data[['Height', 'Weight']].to_numpy()
        labels = data[['Gender']].to_numpy()

        data_point = dropped_row[['Height', 'Weight']].to_numpy()
        expect = dropped_row[['Gender']].to_numpy()
        expected_prediction = np.array(expect[:, -1])

        result = knn_algorithm(feature, labels, data_point, k)
        if result != expected_prediction:
            error_count = error_count + 1
    percent_of_error = error_count / (loop_var - 1) * 100

    return percent_of_error


def main():
    K = [1, 3, 5]
    j = 0
    percent_of_error = [0, 0, 0]
    for k in K:
        percent_of_error[j] = leave_one_out_evaluation(k)
        print("Percent of Error when K =", k, "=>", percent_of_error[j])
        j = j + 1


if __name__ == '__main__':
    main()
