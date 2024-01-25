import numpy as np
from .Layer import Layer


class KfoldWarpper:
    def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, step=3):
        self.num_forests = num_forests  # 2
        self.n_estimators = n_estimators  # 40
        self.n_fold = n_fold # 5
        self.kf = kf  # kf
        self.layer_index = layer_index  # 0
        self.step = step  # 3
        self.model = []

    def train(self, train_data, train_label):
        self.num_labels = train_label.shape[1] # 标签个数 5
        num_samples, num_features = train_data.shape # 样本个数 1000，样本特征 294

        # Gt
        prob = np.empty([num_samples, self.num_labels])
        # Ht
        prob_concatenate = np.empty([self.num_forests, num_samples, self.num_labels])

        fold = 0
        # 训练集 80%，测试集(验证集) 20%，测试集用于预测
        for train_index, test_index in self.kf:
            X_train = train_data[train_index, :]
            X_val = train_data[test_index, :]
            y_train = train_label[train_index, :]

            # training fold-th layer
            layer = Layer(self.n_estimators, self.num_forests, self.num_labels, self.step, self.layer_index,
                          fold)
            layer.train(X_train, y_train)
            self.model.append(layer)
            fold += 1
            prob[test_index], prob_concatenate[:, test_index, :] = layer.predict(X_val)
        return [prob, prob_concatenate]

    def predict(self, test_data):
        test_prob = np.zeros([test_data.shape[0], self.num_labels])
        test_prob_concatenate = np.zeros([self.num_forests, test_data.shape[0], self.num_labels])
        # k-fold交叉
        for layer in self.model:
            temp_prob, temp_prob_concatenate = layer.predict(test_data)
            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold
        return [test_prob, test_prob_concatenate]

    def train_and_predict(self, train_data, train_label, test_data):
        prob, prob_concatenate = self.train(train_data, train_label)
        test_prob, test_prob_concatenate = self.predict(test_data)
        return [prob, prob_concatenate, test_prob, test_prob_concatenate]
