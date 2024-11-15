import subprocess

from KNN import KNNClassifier
from utils import *
from itertools import combinations

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :return: indices of top 'b' features, sorted.
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    top_b_features_indices = []

    # ====== YOUR CODE: ======
    n_samples, n_features = x.shape
    knn_model = KNNClassifier(k=k)

    # bottom-up
    unused_features = list(range(n_features))
    bottom_up_features = []
    while len(bottom_up_features) < b:
        accuracies = []
        for f in unused_features:
            cur_features = bottom_up_features + [f]  # add unused feature
            knn_model.train(x[:, cur_features], y)  # train with new feature
            y_pred = knn_model.predict(x[:, cur_features])  # predict with new feature
            accuracies.append(accuracy(y, y_pred))  # calc acc with new feature

        # find best feature to add
        best_feature_to_add = unused_features.pop(np.argmax(accuracies))
        bottom_up_features.append(best_feature_to_add)

    # top-down
    top_down_features = list(range(n_features))
    while len(bottom_up_features) > b:
        accuracies = []
        for f in top_down_features:
            # remove feature from current features list
            cur_features = top_down_features[:]
            cur_features.remove(f)

            knn_model.train(x[:, cur_features], y_train)  # train wihtout feature
            y_pred = knn_model.predict(x[:, cur_features])  # predict without feature
            accuracies.append(accuracy(y, y_pred))  # calc acc without feature

        # remove feature which removing leads to the highest acc
        top_down_features.pop(np.argmax(accuracies))

    # brute force on best found features (max is 2b features to check)
    best_features_set = set(bottom_up_features) | set(top_down_features)
    best_acc = 0
    for subset in combinations(best_features_set, b):
        cur_features = list(subset)
        knn_model.train(x[:, cur_features], y_train)  # train with subset
        y_pred = knn_model.predict(x[:, cur_features])  # predict with subset
        acc = accuracy(y, y_pred)  # calc acc without feature

        if acc > best_acc:
            best_acc = acc
            top_b_features_indices = cur_features
    # ========================

    return top_b_features_indices


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    # run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 2

    # # ========================================================================

    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, b=b, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
