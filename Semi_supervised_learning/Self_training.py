import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings##忽略警告
warnings.filterwarnings('ignore')

train_feat = pd.read_csv('../data/cache0721/train.csv', header=None)
test_feat = pd.read_csv('../data/cache0721/testA.csv', header=None)
label = pd.read_csv('../data/cache0721/label.txt', header=None)

test_feat_original = test_feat
test_id_original = test_feat_original.ix[:, 0]
test_feat_original = test_feat_original.ix[:, 1:]

# compare to s9982   99.82%
df_9982 = pd.read_csv("../data/his_submit/s9982.txt",header=None)
df_9982.columns = ['id']
print(len(df_9982))

train_feat = train_feat.ix[:, 1:]
test_id = test_feat.ix[:, 0]
test_feat = test_feat.ix[:, 1:]
x_train, x_val, y_train, y_val = train_test_split(train_feat, label, test_size=0.05, random_state=518)
clf = xgb.XGBClassifier(max_depth=6,
                        learning_rate=0.05,
                        n_estimators=2000,
                        objective='reg:logistic',
                        gamma=0.0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=1,
                        colsample_bytree=0.8,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1
                        )
clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
y_submission = clf.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)
pred[['id', 'prob']].to_csv('round_1_prediction_all.csv', index=False)
pred1 = pred.iloc[:20000]
right = len(pred1[pred1.id.isin(df_9982.id.unique())])
print('round 0 right number:{},score:{}'.format(right, right * 100.00 / 20000))

# 取前n个后m个打标加入L
######################################self-training#########################################
n = 12000#前n个正例
m = 36000#后m个反例
T = 5#循环轮数
for num in range(1, T + 1):
    add_true = pred.ix[0:n - 1, 2:]
    add_false = pred.ix[pred.shape[0] - m:, 2:]
    pred_id_prob = pred.ix[n:pred.shape[0]-m, :]
    pred = pred.ix[n:pred.shape[0]-m, 2:]  ###剔除选取的样本
    if pred.shape[0] == 0:
        print("all labeled,stop at %d" % (num))
        break

    add_label = np.array([0] * n + [1] * m).reshape((n+m, 1))
    train_feat_add = np.concatenate((train_feat, add_true.values, add_false.values), axis=0)
    label_add = np.concatenate((label, add_label), axis=0)
    train_feat = train_feat_add
    label = label_add
    test_feat = pred
    test_id = pred_id_prob.ix[:, 0]

    x_train, x_val, y_train, y_val = train_test_split(train_feat, label, test_size=0.1, random_state=518)
    idx = np.random.permutation(label.size)
    train_feat_add = train_feat_add[idx]
    label_add = label_add[idx]

    clf = xgb.XGBClassifier(max_depth=5,
                            learning_rate=0.05,
                            n_estimators=1500,
                            objective='reg:logistic',
                            gamma=0.0,
                            min_child_weight=0.8,
                            max_delta_step=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            colsample_bylevel=1,
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1
                            )
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
    y_submission = clf.predict_proba(test_feat.values)[:, 0]
    pred = {'id': test_id, 'prob': y_submission}
    pred = pd.DataFrame.from_dict(pred)
    pred = pd.concat([pred, test_feat], axis=1)
    pred.sort_values(by='prob', ascending=[0], inplace=True)
    pred = pred.reset_index(drop=True)
    #############每轮训练的模型xgb对初始test_feat预测，取前两万个与s99.82做比较，打分，即每轮训练xgb的性能
    y_submission = clf.predict_proba(test_feat_original.values)[:, 0]
    pred_original = {'id': test_id_original, 'prob': y_submission}
    pred_original = pd.DataFrame.from_dict(pred_original)
    pred_original = pd.concat([pred_original, test_feat_original], axis=1)
    pred_original.sort_values(by='prob', ascending=[0], inplace=True)
    pred_original = pred_original.reset_index(drop=True)
    pred1 = pred_original.iloc[:20000]
    right = len(pred1[pred1.id.isin(df_9982.id.unique())])
    print('round %d' % (num))
    print(' right number:{},score:{}'.format(right, right * 100.00 / 20000))


