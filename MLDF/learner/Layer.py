import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier


class Layer:
    def __init__(self, n_estimators, num_forests, num_labels, step=3, layer_index=0, fold=0):
        self.n_estimators = n_estimators # 每个森林树的个数 40
        self.num_labels = num_labels
        self.num_forests = num_forests
        self.layer_index = layer_index
        self.fold = fold
        self.step = step
        self.model = []

    def train(self, train_data, train_label):
        n_estimators = min(20 * self.layer_index + self.n_estimators, 100)
        max_depth = self.step * self.layer_index + self.step
        for forest_index in range(self.num_forests):
            if forest_index % 2 == 0:
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion="gini",
                                             max_depth=max_depth,
                                             n_jobs=-1,
                                             random_state=520
                                             )
            else:
                clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                           criterion="gini",
                                           max_depth=max_depth,
                                           n_jobs=-1,
                                           random_state=1
                                           )
            has_nan = np.isnan(train_data).any()
            has_inf = np.isinf(train_data).any()
            if has_nan or has_inf:
                train_data = np.nan_to_num(train_data, nan=0, posinf=0, neginf=0)
            train_data = train_data.astype(np.float32)
            has_nan1 = np.isnan(train_label).any()
            has_inf1 = np.isinf(train_label).any()
            if has_nan1 or has_inf1:
                train_label = np.nan_to_num(train_label, nan=0, posinf=0, neginf=0)
            train_label = train_label.astype(np.float32)
            clf.fit(train_data, train_label)
            self.model.append(clf)
        self.layer_index += 1

    def predict(self, test_data):
        predict_prob = np.zeros(
            [self.num_forests, test_data.shape[0], self.num_labels])

        for forest_index, clf in enumerate(self.model):
            has_nan = np.isnan(test_data).any()
            has_inf = np.isinf(test_data).any()
            if has_nan or has_inf:
                test_data = np.nan_to_num(test_data, nan=0, posinf=0, neginf=0)
            # 每个标签的预测是(1*2)的数组，[0的概率，1的概率]
            predict_p = clf.predict_proba(test_data)
            for j in range(len(predict_p)):
                predict_prob[forest_index, :, j] = 1 - predict_p[j][:, 0].T

        # 先求和两个森林的结果，再平均每个森林的结果，得到每个样本的每个标签的概率
        prob_avg = np.sum(predict_prob, axis=0)
        prob_avg /= self.num_forests
        prob_concatenate = predict_prob
        return [prob_avg, prob_concatenate]

    def train_and_predict(self, train_data, train_label, val_data, test_data):
        self.train(train_data, train_label)
        val_avg, val_concatenate = self.predict(val_data)
        prob_avg, prob_concatenate = self.predict(test_data)

        return [val_avg, val_concatenate, prob_avg, prob_concatenate]
