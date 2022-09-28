
import numpy as np
import pandas as pd

def load_data():
    data = pd.read_csv('data_2c.csv')

    labeled_data = data.iloc[:20]
    unlabeled_data_with_label = data.iloc[20:]

    unlabeled_data = unlabeled_data_with_label.iloc[:, :-1]
    original_label_of_unlabeled_data = unlabeled_data_with_label.iloc[:, -1]
    return labeled_data, unlabeled_data, original_label_of_unlabeled_data

def cartesian_distance(feature, converted_array_of_data_point):
    distance_matrix = np.power((feature - converted_array_of_data_point), 2)

    return np.sqrt(np.sum(distance_matrix, axis=1))

def self_training(labeled_data, unlabeled_data, number_of_data):

    total_unlabeled_data_count = unlabeled_data.shape[0]
    labeled_data_count = 0
    correct = 0

    while total_unlabeled_data_count > labeled_data_count:
        feature = labeled_data[['Height', 'Weight', 'Age']].to_numpy()
        labels = labeled_data[['Gender']].to_numpy()
        temp_data_items = unlabeled_data.copy()
        temp_data_items['Gender'] = ''
        temp_data_items['Certainty'] = ''
        for ind in temp_data_items.index:
            data_point =temp_data_items.loc[[ind]]
            data_point = data_point[['Height', 'Weight', 'Age']].values.reshape(-1)
            gender,weight = knn_algorithm(feature, labels, data_point,5)
            temp_data_items.at[ind,"Gender"] = gender
            temp_data_items.at[ind,"Certainty"] = weight
        sorted_data_items = temp_data_items.sort_values("Certainty", ascending=False)
        sorted_data_items = sorted_data_items.iloc[:number_of_data]
        sorted_data_items_index = list(sorted_data_items.index)
        for ind in sorted_data_items_index:
                labeled_data = pd.concat([labeled_data,  temp_data_items.loc[[ind]]], axis=0, join='outer',ignore_index=True)
                unlabeled_data = unlabeled_data.drop(ind)
        labeled_data_count = labeled_data_count + number_of_data
    
    labeled_data = labeled_data.drop(['Certainty'], axis = 1)
    return  labeled_data

def knn_algorithm(feature, labels, data_point, k):
    row = feature.shape[0]
    collum = feature.shape[1]
    weighted_vote_men = 0
    weighted_vote_women = 0
    converted_array_of_data_point = np.full((row, collum), data_point)
    
    distance = cartesian_distance(feature, converted_array_of_data_point)

    distance_labels = np.column_stack((distance, labels))

    distance_labels = distance_labels[np.argsort(distance_labels[:, 0])]
    k_nearest_points = distance_labels[:k, :]

    for i in range(k):
        if k_nearest_points[i,0] != 0:
            if k_nearest_points[i,1] == 'M' :
                weighted_vote_men = weighted_vote_men + (1/k_nearest_points[i,0])
            else:
                weighted_vote_women = weighted_vote_women + (1/k_nearest_points[i,0])
        else:
            if k_nearest_points[i,1] == 'M' :
                weighted_vote_men = weighted_vote_men + 1
            else:
                weighted_vote_women = weighted_vote_women + 1 

    if weighted_vote_men > weighted_vote_women :
        result,weight = str("M"),weighted_vote_men
    else:
        result,weight = str("W"),weighted_vote_women
    return result,weight



def compare(learned_classifier, labeled_data, unlabeled_data,original_label_of_unlabeled_data):
    correct_KNN20 = 0
    correct_selftrain = 0
    feature_KNN20 = labeled_data[['Height', 'Weight', 'Age']].to_numpy()
    labels_KNN20 = labeled_data[['Gender']].to_numpy()
    feature_selftrain = learned_classifier[['Height', 'Weight', 'Age']].to_numpy()
    labels_selftrain = learned_classifier[['Gender']].to_numpy()

    for ind in unlabeled_data.index:
          data_point = unlabeled_data.loc[[ind]].values.reshape(-1)
          gender_KNN20 ,waste = knn_algorithm(feature_KNN20, labels_KNN20, data_point, 5)
          if original_label_of_unlabeled_data.loc[[ind]].values == gender_KNN20:
              correct_KNN20 = correct_KNN20 + 1
          gender_selftrain, waste = knn_algorithm(feature_selftrain, labels_selftrain, data_point, 5)
          if original_label_of_unlabeled_data.loc[[ind]].values == gender_selftrain:
              correct_selftrain = correct_selftrain + 1
    accuracy_KNN20 = (correct_KNN20/unlabeled_data.shape[0])*100
    accuracy_selftrain = (correct_selftrain/unlabeled_data.shape[0])*100
    return accuracy_KNN20, accuracy_selftrain

def main():
    labeled_data, unlabeled_data, original_label_of_unlabeled_data = load_data()

 
    for i in [1, 5, 25,unlabeled_data.shape[0]]:
        
        learned_classifier = self_training(labeled_data, unlabeled_data, i)
        accuracy_KNN20, accuracy_selftrain = compare(learned_classifier, labeled_data, unlabeled_data,original_label_of_unlabeled_data)
        print('\033[1m' + 'Correctly predicted label using first 20 data points with KNN:',accuracy_KNN20, ' '+ '\033[0m')
        print('\033[1m' + 'Correctly predicted label using ',i,' point per itaration Self-Training:',accuracy_selftrain, ' '+ '\033[0m')

if __name__ == '__main__':
    main()