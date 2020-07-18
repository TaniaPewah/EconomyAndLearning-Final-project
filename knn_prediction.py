# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
import numpy as np
from csv import reader
from math import sqrt
import pandas as pd


# Load a CSV file
def load_csv(filename, no_first):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for indx, row in enumerate(csv_reader):
            if no_first and indx == 0:
                continue
            if not row:
                continue
            dataset.append(row)
    return dataset

def cut_first_col(dataset):
    for indx, row in enumerate(dataset):
        dataset[indx] = row[1:]


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [float(row[i]) for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(1, len(row)-1):
            if (minmax[i][1] - minmax[i][0]) == 0:
                continue
            row[i] = (float(row[i]) - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1, len(row1)-1):
        distance += (float(row1[i]) - float(row2[i]))**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    if test_row[1] == 1:
        train = [trial for trial in train if trial[1] == 1]
    elif test_row[1] == 0:
        train = [trial for trial in train if trial[1] == 0]

    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]

    output_values = [float(val) for val in output_values]
    prediction = np.mean(output_values)
    #prediction = max(set(output_values), key=output_values.count)
    return prediction


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        # (self.a_block_percentage[i] - self.actual_A[i])**2
        correct += (predicted[i] - float(actual[i]))**2
    return (1 - (correct / float(len(actual)))) * 100.0

#######################################
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, num_neighbours, col_name):

    cros_fold_predictions = np.zeros(len(dataset))

    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for i, fold in enumerate(folds):
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, num_neighbours)

        for test_i, trial in enumerate(test_set):
            print("trial: ", trial[0] , " fold: ", i, " predicted: ", predicted[test_i])
            cros_fold_predictions[int(trial[0])-1] = predicted[test_i]

        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    print("cros_fold_predictions: ", cros_fold_predictions)
    add_col_to_csv(cros_fold_predictions, col_name)

    return scores

def add_col_to_csv(cros_fold_predictions, col_name):
    df = pd.read_csv("output_knn.csv")
    df[col_name] = cros_fold_predictions
    df.to_csv("output_knn.csv", index=False)


# Test the kNN on the Iris Flowers dataset
seed(1)

# evaluate algorithm
n_folds = 4
num_neighbors = 2
for block in range(0,5):
    blockname = 'A'+ str(block+1)
    filename = 'train' + blockname + '.csv'
    no_first = True
    dataset = load_csv(filename, no_first)


    normalize_dataset(dataset, dataset_minmax(dataset))
    #cut_first_col(dataset)

    scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors, blockname)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

for block in range(0, 5):
    blockname = 'B' + str(block + 1)
    filename = 'train' + blockname + '.csv'
    no_first = True
    dataset = load_csv(filename, no_first)

    normalize_dataset(dataset, dataset_minmax(dataset))

    scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors, blockname)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


