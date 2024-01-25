from sklearn.utils import shuffle

from learner.cascade import Cascade
from learner.measure import *


def shuffle_index(num_samples):
    a = range(0, num_samples)
    a = shuffle(a)
    length = int((num_samples + 1) / 2)
    train_index = a[:length]
    test_index = a[length:]
    return [train_index, test_index]


def make_data(dataset):
    data = np.load("dataset/{}_data.npy".format(dataset))
    label = np.load("dataset/{}_label.npy".format(dataset))
    print(data.shape)
    print(label.shape)
    data = data.astype(np.float32)
    label = label.astype("int")
    num_samples = data.shape[0]
    train_index, test_index = shuffle_index(num_samples)
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data[test_index]
    test_label = label[test_index]
    return [train_data, train_label, test_data, test_label]


if __name__ == '__main__':
    dataset = "image"
    # dataset = "scene"
    # dataset = "enron"
    # dataset = "yeast"
    # dataset = "mediamill"
    # dataset = "CAL500"
    # dataset = "eurlex-sm"
    # dataset = "corel16k-s1"

    # train_data, train_label, test_data, test_label = make_data(dataset)
    # model = Cascade(dataset, max_layer=10, num_forests=2, n_fold=5, step=3)
    # model.train(train_data, train_label, "macro_auc", n_estimators=40)
    # test_prob = model.predict(test_data, "macro_auc")
    # value = do_metric(test_prob, test_label, 0.5)
    # print("hamming loss")
    # print(["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"])
    # print(value)

    # has_nan = np.isnan(train_data).any()
    # has_inf = np.isinf(train_data).any()
    # if has_nan:
    #     print("train_data 中包含 NaN 值")
    # if has_inf:
    #     print("train_data 中包含 infinity 值")

    epoch = 5
    all_value = []
    for i in range(1, epoch + 1):
        train_data, train_label, test_data, test_label = make_data(dataset)
        model = Cascade(dataset, max_layer=10, num_forests=2, n_fold=5, step=3)
        model.train(train_data, train_label, "one-error", n_estimators=40)
        test_prob = model.predict(test_data, "one-error")
        value = do_metric(test_prob, test_label, 0.5)
        all_value.append(value)
    print(["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"])
    array = np.array(all_value)
    print(array)
    all_average = np.mean(array, axis=0)
    print("average: ", all_average)
    variances = np.var(array, axis=0)
    print("variances: ", variances)
