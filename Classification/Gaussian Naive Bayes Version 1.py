import numpy as np
import pandas as pd
import math


def load_data():
    data = pd.read_csv('data_2b.csv')
    data[['Height', 'Weight', 'Age']] = data[['Height', 'Weight', 'Age']].apply(pd.to_numeric)
    return data


def load_test_data():
    test_data = pd.read_csv('test_data.csv')

    return test_data


def mean_variance(df):
    data_mean = df.groupby('Gender').mean()

    mean = data_mean[['Height', 'Weight', 'Age']].to_numpy()

    data_var = df.groupby('Gender').var()
    var = data_var[['Height', 'Weight', 'Age']].to_numpy()
    return mean,var


def gussian_pdf(x, mean, variance):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * variance)))

    return (1 / (math.sqrt(2 * math.pi * variance))) * exponent


def prior_probability(data):
    data_in_numarr = data[['Height', 'Weight', 'Age', 'Gender']].to_numpy()
    class_labels = np.array(data_in_numarr[:, -1])
    no_of_men_class, no_of_women_class = np.count_nonzero(class_labels == 'M'), np.count_nonzero(class_labels == 'W')
    total_classes = no_of_women_class + no_of_men_class
    prior_men, prior_women = no_of_men_class / total_classes, no_of_women_class / total_classes
    prior = np.array([prior_men, prior_women])
    return prior


def likelihood_cal(mean, variance, test_data):
    number_of_test_data = test_data.shape[0]
    no_feature = test_data.shape[1]

    likelihood_prob = np.zeros((number_of_test_data, 2))

    for t in range(0, number_of_test_data):

        temp = test_data[t]

        tp = np.zeros(2)
        for i in range(0, 2):

            mul = 1
            for j in range(0, no_feature):
                mul = mul * gussian_pdf(temp[j], mean[i][j], variance[i][j])
            tp[i] = mul

        likelihood_prob[t] = tp

    return likelihood_prob


def posterior_probability(prior, likelihood):
    posterior_prob = np.ones(2)
    total_prob = 0
    for i in range(0, 2):
        total_probability = total_prob + (prior[i] * likelihood[i])

    for i in range(0, 2):
        posterior_prob[i] = (prior[i] * likelihood[i]) / total_probability

    return posterior_prob


def Gaussian_Naive_Bayes(data, test_data):

    mean, variance = mean_variance(data)

    prior = prior_probability(data)

    likelihood = likelihood_cal(mean, variance, test_data)

    for i in range(0, likelihood.shape[0]):

        print('Test data: ', test_data[i])

        posterior_prob = posterior_probability(prior, likelihood[i])
        predict = int(posterior_prob.argmax())
        print("Probability ditribution for two class :\n[M,W] = ", posterior_prob)
        if predict == 0:
            print("Predicted Gender: M")
        else:
            print("Predicted Gender: W")


def main():
    data = load_data()

    test_data1 = load_test_data()

    test_data = test_data1[['Height', 'Weight', 'Age']].to_numpy()

    Gaussian_Naive_Bayes(data, test_data)


if __name__ == '__main__':
    main()
