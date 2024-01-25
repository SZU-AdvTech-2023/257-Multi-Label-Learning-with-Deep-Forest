from sklearn.cross_validation import KFold
from .measure import *
from .warpper import KfoldWarpper


class Cascade:
    def __init__(self, dataname, max_layer=20, num_forests=2, n_fold=5, step=3):
        self.max_layer = max_layer
        self.n_fold = n_fold
        self.step = step
        self.layer_list = []
        self.num_forests = num_forests
        self.dataname = dataname
        self.eta = []
        self.model = []

    def compute_confidence(self, supervise, P):
        """
        :param supervise: string (e.g. "hamming loss", "one-error")
        :param P: array, whose shape is (num_samples, num_labels)
        :return alpha: array, whose shape is (num_samples, ) when supervise is instance-based measure,
                        and (num_labels, ) when supervise is label-based measure
        """
        m, l = P.shape
        if supervise == "hamming loss":
            alpha = np.sum(np.abs(P - 0.5) + 0.5, axis=0) / m
        elif supervise == "one-error":
            alpha = np.max(P, axis=1)
        elif supervise == "ranking loss" or supervise == "average precision":
            forward_prod = np.sort(P, axis=1)
            backward_prod = 1 - forward_prod
            for j in range(1, l, 1):
                forward_prod[:, j] = forward_prod[:, j - 1] * P[:, j]
            for j in range(l - 2, -1, -1):
                backward_prod[:, j] = backward_prod[:, j + 1] * (1 - P[:, j])
            alpha = forward_prod[:, l - 1] + backward_prod[:, 0]
            for j in range(l - 1):
                alpha += forward_prod[:, j] * backward_prod[:, j + 1]
        elif supervise == "coverage":
            backward_prod = 1 - np.sort(P, axis=1)
            for j in range(l - 2, -1, -1):
                backward_prod[:, j] = backward_prod[:, j + 1] * (1 - P[:, j])
            alpha = backward_prod[:, 0]
            for j in range(l - 1):
                alpha += j * P[:, j] * backward_prod[:, j + 1]
            alpha = 1 - alpha / l
        elif supervise == "macro_auc":
            forward_prod = np.sort(P, axis=0)
            backward_prod = 1 - P.copy()
            for i in range(1, m, 1):
                forward_prod[i, :] = forward_prod[i - 1, :] * P[i, :]
            for i in range(m - 2, -1, -1):
                backward_prod[i, :] = backward_prod[i + 1, :] * (1 - P[i, :])
            alpha = forward_prod[m - 1, :] + backward_prod[0, :]
            for i in range(m - 1):
                alpha += forward_prod[i, :] * backward_prod[i + 1, :]
        return alpha

    def train(self, train_data_raw, train_label_raw, supervise, n_estimators=40):
        """
        :param train_data_raw: array, whose shape is (num_samples, num_features)
        :param train_label_raw: array, whose shape is (num_samples, num_labels)
        :param supervise: string, (e.g. "hamming loss", "one-error")
        :param n_estimators: int, num_trees in each forest
        """
        train_data = train_data_raw.copy()
        train_label = train_label_raw.copy()
        self.num_labels = train_label.shape[1]
        best_value = init_supervise(supervise)
        bad = 0

        best_train_prob = np.empty(train_label.shape)
        best_concatenate_prob = np.empty([self.num_forests, train_data.shape[0], self.num_labels])
        for layer_index in range(self.max_layer):
            print("training layer " + str(layer_index))
            kf = KFold(len(train_label), n_folds=self.n_fold, shuffle=True)
            kfoldwarpper = KfoldWarpper(
                self.num_forests, n_estimators, self.n_fold, kf, layer_index, self.step)
            prob, prob_concatenate = kfoldwarpper.train(train_data, train_label)
            self.model.append(kfoldwarpper)
            if layer_index == 0:
                best_train_prob = prob
                pre_metric = compute_supervise_vec(
                    supervise, best_train_prob, train_label, 0.5)
            else:
                now_metric = compute_supervise_vec(
                    supervise, prob, train_label, 0.5)
                if supervise == "average precision" or supervise == "macro_auc":
                    indicator = now_metric < pre_metric
                else:
                    indicator = now_metric > pre_metric
                if np.nansum(indicator) > 0:
                    confidence = self.compute_confidence(supervise, prob)
                    eta_t = np.nanmean(confidence[indicator])
                    # eta_t = 0
                    train_indicator = confidence < eta_t
                    if supervise == "hamming loss" or supervise == "macro_auc":
                        prob[:, train_indicator] = best_train_prob[:, train_indicator]
                        prob_concatenate[:, :, train_indicator] = best_concatenate_prob[:, :,
                                                                  train_indicator]
                    else:
                        prob[train_indicator, :] = best_train_prob[train_indicator, :]
                        prob_concatenate[:, train_indicator, :] = best_concatenate_prob[:, train_indicator,
                                                                  :]
                else:
                    eta_t = 0
                self.eta.append(eta_t)
                best_train_prob = prob
                best_concatenate_prob = prob_concatenate
                pre_metric = compute_supervise_vec(supervise, best_train_prob, train_label, 0.5)
            value = compute_supervise(supervise, best_train_prob, train_label, 0.5)
            back = compare_supervise_value(supervise, best_value, value)
            if back:
                bad += 1
            else:
                bad = 0
                best_value = value
            if bad >= 3:
                for i in range(bad):
                    self.model.pop()
                    self.eta.pop()
                break
            # prepare data of next layer
            prob_concatenate = best_concatenate_prob.transpose((1, 0, 2))
            prob_concatenate = prob_concatenate.reshape(prob_concatenate.shape[0], -1)
            train_data = np.concatenate([train_data_raw.copy(), prob_concatenate], axis=1)


    def predict(self, test_data_raw, supervise):
        """
        :param test_data_raw: array, whose shape is (num_test_samples, num_features)
        :return prob: array, whose shape is (num_test_samples, num_labels)
        """
        test_data = test_data_raw.copy()
        best_prob = np.empty([test_data.shape[0], self.num_labels])
        best_concatenate_prob = np.empty([self.num_forests, test_data.shape[0], self.num_labels])
        for clf, eta_t in zip(self.model, self.eta):
            prob, prob_concatenate = clf.predict(test_data)
            confidence = self.compute_confidence(supervise, prob)
            indicator = confidence < eta_t
            if supervise == "hamming loss" or supervise == "macro_auc":
                prob[:, indicator] = best_prob[:, indicator]
                prob_concatenate[:, :, indicator] = best_concatenate_prob[:, :, indicator]
            else:
                prob[indicator, :] = best_prob[indicator, :]
                prob_concatenate[:, indicator, :] = best_concatenate_prob[:, indicator, :]
            best_concatenate_prob = prob_concatenate
            best_prob = prob
            prob_concatenate = best_concatenate_prob.transpose((1, 0, 2))
            prob_concatenate = prob_concatenate.reshape(prob_concatenate.shape[0], -1)
            test_data = np.concatenate([test_data_raw.copy(), prob_concatenate], axis=1)
        return best_prob
