from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# generate two class dataset
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20, random_state=27)

# split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)


# train models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# logistic regression
model1 = LogisticRegression()
# knn
model2 = KNeighborsClassifier(n_neighbors=4)

# # fit model
# model1.fit(X_train, y_train)
# model2.fit(X_train, y_train)
#
# # predict probabilities
# pred_prob1 = model1.predict_proba(X_test)
# pred_prob2 = model2.predict_proba(X_test)
#
#
# print(pred_prob1[:10])
# print(pred_prob1[:,1][:10])
# print(y_test)
#
# from sklearn.metrics import roc_curve
#
#
# #
# # # roc curve for tpr = fpr
# # random_probs = [0 for i in range(len(y_test))]
# # p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
# #
# from sklearn.metrics import roc_auc_score
#
# # auc scores
# auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
# auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
#
# print(auc_score1, auc_score2)

import torch
import torch.nn as nn

def group_norn(x: torch.Tensor,
               num_groups: int,
               num_channels: int,
               eps: float = 1e-5,
               gamma: float = 1.0,
               beta: float=0.):
    assert divmod(num_channels, num_channels)[1] == 0
    channels_per_group = num_channels // num_groups

    new_tensor = []
    for t in x.split(channels_per_group, dim=1):
        print(t.size())
        var_mean = torch.var_mean(t, dim=[1, 2, 3], unbiased=False)
        var = var_mean[0]
        mean = var_mean[1]
        t = (t - mean[:, None, None, None]) / torch.sqrt(var[:, None, None, None] + eps)
        t = t * gamma + beta
        new_tensor.append(t)

    new_tensor = torch.cat(new_tensor, dim=1)
    return new_tensor


def main():

    num_groups = 2
    num_channels = 4
    eps = 1e-5
    img = torch.rand(2, num_channels, 2, 2)
    print(img)

    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)

    r1 = gn(img)
    print(r1)

    r2 = group_norn(img, num_groups, num_channels, eps)
    print(r2)



if __name__ == '__main__':
    main()
