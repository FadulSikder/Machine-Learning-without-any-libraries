import numpy as np
import pandas as pd
import math


def load_data():
    data = pd.read_csv('data_2c.csv')
    return data


def mean_variance(df):
    data_mean = df.groupby('Gender').mean()

    mean = data_mean[['Height', 'Weight', 'Age']].to_numpy()

    data_var = df.groupby('Gender').var()
    var = data_var[['Height', 'Weight', 'Age']].to_numpy()
    return mean, var


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

        pp = np.zeros(2)
        for i in range(0, 2):

            mul = 1

            for j in range(0, no_feature):
                mul = mul * gussian_pdf(temp[j], mean[i][j], variance[i][j])
            pp[i] = mul

        likelihood_prob[t] = pp

    return likelihood_prob


def posterior_probability(prior, likelihood):
    posterior_prob = np.ones(2)
    total_prob = 0
    for i in range(0, 2):
        total_probability= total_prob + (prior[i] * likelihood[i])

    for i in range(0, 2):
        posterior_prob[i] = (prior[i] * likelihood[i]) / total_probability

    return posterior_prob


def Gaussian_Naive_Bayes(data, test_data):

    mean, variance = mean_variance(data)

    prior = prior_probability(data)

    likelihood = likelihood_cal(mean, variance, test_data)

    for i in range(0, likelihood.shape[0]):
        posterior_prob = posterior_probability(prior, likelihood[i])
        predict = int(posterior_prob.argmax())


    return 'M' if predict == 0 else 'W'


def leave_one_out_evaluation():
    pre_data = load_data()

    loop_var = len(pre_data.index)
    error_count = 0

    for i in range(loop_var):

        dropped_row = pre_data.iloc[[i], :]
        data = pre_data.drop(pre_data.index[i])
        test_data = dropped_row[['Height', 'Weight', 'Age']].to_numpy()

        expect = dropped_row[['Gender']].to_numpy()
        expected_prediction = np.array(expect[:, -1])
        result = Gaussian_Naive_Bayes(data, test_data)
        if result != expected_prediction:
            error_count = error_count + 1
    percent_of_error = error_count / loop_var * 100

    return percent_of_error


def main():
    percent_of_error = leave_one_out_evaluation()
    print("Percent of Error =>", percent_of_error)


if __name__ == '__main__':
    main()
