import numpy as np
import pandas as pd


def load_data():
    data = pd.read_csv('data_2b.csv')
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

    result = 'M' if men_occurencies > women_occurencies else 'W'

    print("Result for K = ", k, "\nPredicted Gender : ", result)


def take_input():
    data_point = list(map(float, input("\nEnter your data (Height Weight Age) : ").strip().split()))[:3]

    return np.asarray(data_point)


def main():
    pre_data = load_data()

    feature = pre_data[['Height', 'Weight', 'Age']].to_numpy()

    labels = pre_data[['Gender']].to_numpy()

    K = [1, 3, 5]

    data_point = take_input()

    for k in K:
        knn_algorithm(feature, labels, data_point, k)


if __name__ == '__main__':
    main()
