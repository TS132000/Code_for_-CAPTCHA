######导入数据
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings  ##忽略警告

warnings.filterwarnings('ignore')


train_feat = pd.read_csv('../data/cache0721/train.csv', header=None)
test_feat = pd.read_csv('../data/cache0721/testA.csv', header=None)
label = pd.read_csv('../data/cache0721/label.txt', header=None)
test_feat_original = test_feat
test_id_original = test_feat_original.ix[:, 0]
test_feat_original = test_feat_original.ix[:, 1:]

# compare to s9982   99.82%
df_9982 = pd.read_csv("../data/his_submit/s9982.txt", header=None)
df_9982.columns = ['id']
print(len(df_9982))

train_feat = train_feat.ix[:, 1:]
test_id = test_feat.ix[:, 0]
test_feat_original = test_feat
test_feat = test_feat.ix[:, 1:]

### create U_
U = test_feat_original
test_feat_original[0] = test_feat_original[0] - 1
U_size = 10000
U_ = U.sample(n=U_size, random_state=123, axis=0)
U_id = U_.iloc[:, :1]
Uid = U.iloc[:, :0]
id = np.array(U_id)
# U 剔除 U'
for i in range(0, 100000):
    if i in id:
        U = U.drop(i, axis=0)
    else:
        continue

U_ = U_.reset_index(drop=True)

#### 试验一：前35为视图一，后35为视图二###########
train_feat_view1 = train_feat.iloc[:, :35]
train_feat_view2 = train_feat.iloc[:, 35:70]
train_feat_view1_original = train_feat_view1
train_feat_view2_original = train_feat_view2
#############缓冲池视图划分###########
U_view1 = U_.iloc[:, :36]
U_view2 = U_.iloc[:, 36:]
U_view1_original = U_view1
U_view2_original = U_view2
U_view1_test = U_view1.ix[:, 1:]
U_view2_test = U_view2
###############test_feat视图划分#########################3
test_feat_view1 = test_feat.iloc[:, :35]
test_feat_view2 = test_feat.iloc[:, 35:70]
test_feat_view1_original = test_feat_view1
test_feat_view2_original = test_feat_view2
##########################注意！！
test_id = test_id + 1
# 训练两个分类器clf1，clf2
x_train, x_val, y_train, y_val = train_test_split(train_feat_view1, label, test_size=0.05, random_state=518)
clf1 = xgb.XGBClassifier(max_depth=6,
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
clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
##################################基于原10万条数据视图1部分，评价clf1性能##########################
y_submission1 = clf1.predict_proba(test_feat_view1)[:, 0]
pred_view1 = {'id': test_id, 'prob': y_submission1}
pred_view1 = pd.DataFrame.from_dict(pred_view1)
pred_view1 = pd.concat([pred_view1, test_feat_view1], axis=1)
pred_view1.sort_values(by='prob', ascending=[0], inplace=True)
pred_view1 = pred_view1.reset_index(drop=True)

pred1 = pred_view1.iloc[:20000]
right = len(pred1[pred1.id.isin(df_9982.id.unique())])
print('round 1 clf1 right number:{},score:{}'.format(right, right * 100.00 / 20000))

x_train, x_val, y_train, y_val = train_test_split(train_feat_view2, label, test_size=0.05, random_state=518)
clf2 = xgb.XGBClassifier(max_depth=6,
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
clf2.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
##################################基于原10万条数据视图2部分，评价clf2性能##########################
y_submission2 = clf2.predict_proba(test_feat_view2)[:, 0]

pred_view2 = {'id': test_id, 'prob': y_submission2}
pred_view2 = pd.DataFrame.from_dict(pred_view2)
pred_view2 = pd.concat([pred_view2, test_feat_view2], axis=1)

pred_view2.sort_values(by='prob', ascending=[0], inplace=True)
pred_view2 = pred_view2.reset_index(drop=True)

pred2 = pred_view2.iloc[:20000]
right = len(pred2[pred2.id.isin(df_9982.id.unique())])
print('round 1 clf2 right number:{},score:{}'.format(right, right * 100.00 / 20000))
test_id_view2 = test_id[:10000]
##############################训练的学习器，对缓冲池中view1部分做预测,并按置信度排序###############
y_submission = clf1.predict_proba(U_view1_test)[:, 0]
pred_U_view1 = {'id': test_id_view2, 'prob': y_submission}
pred_U_view1 = pd.DataFrame.from_dict(pred_U_view1)

pred_U_view1 = pd.concat([pred_U_view1, U_view1_test], axis=1)
pred_U_view1.sort_values(by='prob', ascending=[0], inplace=True)
pred_U_view1 = pred_U_view1.reset_index(drop=True)

pred_U_view1_id = pred_U_view1.ix[:, 0]
id = np.array(pred_U_view1_id)
U_array = np.array(U_)
U_array_add = U_array[id - 1]  ######view2
U_array_add = pd.DataFrame.from_dict(U_array_add)
# U_view2_feat = U_view2_test_array_add


q = 1000
n = 3000

add_true_view2 = U_array_add.ix[0:q - 1, 36:]
add_false_view2 = U_array_add.ix[U_array_add.shape[0] - n:, 36:]
add_label = np.array([0] * q + [1] * n).reshape((q + n, 1))
train_feat_view2 = np.concatenate((train_feat_view2, add_true_view2.values, add_false_view2.values), axis=0)
label_add_view2 = np.concatenate((label, add_label), axis=0)
U_ = U_.ix[q:U_array_add.shape[0] - n:]  # 剔除

test_id_original = test_id_original + 1
test_id_original
###########注意检查

x_train, x_val, y_train, y_val = train_test_split(train_feat_view2, label_add_view2, test_size=0.1, random_state=518)
clf_view2_1 = xgb.XGBClassifier(max_depth=5,
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
clf_view2_1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
##########################################在最初视图二的10万条数据上做预测，评价xgb性能￥########################

y_submission = clf_view2_1.predict_proba(test_feat_view2_original.values)[:, 0]
pred_original = {'id': test_id_original, 'prob': y_submission}
pred_original = pd.DataFrame.from_dict(pred_original)
pred_original = pd.concat([pred_original, test_feat_original], axis=1)
pred_original.sort_values(by='prob', ascending=[0], inplace=True)
pred_original = pred_original.reset_index(drop=True)
pred1 = pred_original.iloc[:20000]
right = len(pred1[pred1.id.isin(df_9982.id.unique())])
print('round %d，clf_view2_1  ' % (1))
print(' right number:{},score:{}'.format(right, right * 100.00 / 20000))

test_id_view2 = test_id[:6001]
test_id_view2

##############################训练的学习器，对缓冲池中view2部分做预测,并按置信度排序###############
U_view2_test = U_.iloc[:, 36:]
U_view2_test = U_view2_test.reset_index(drop=True)
y_submission = clf2.predict_proba(U_view2_test)[:, 0]
pred_U_view2 = {'id': test_id_view2, 'prob': y_submission}
pred_U_view2 = pd.DataFrame.from_dict(pred_U_view2)

pred_U_view2 = pd.concat([pred_U_view2, U_view2_test], axis=1)
pred_U_view2.sort_values(by='prob', ascending=[0], inplace=True)
pred_U_view2 = pred_U_view2.reset_index(drop=True)

pred_U_view2_id = pred_U_view2.ix[:, 0]
id = np.array(pred_U_view2_id)
U_array = np.array(U_)
U_array_add = U_array[id - 1]  ######view2
U_array_add = pd.DataFrame.from_dict(U_array_add)
# U_view2_feat = U_view2_test_array_add


q = 1000
n = 3000

add_true_view1 = U_array_add.ix[0:q - 1, :35]
add_true_view1 = add_true_view1.ix[:, 1:]
add_false_view1 = U_array_add.ix[U_array_add.shape[0] - n:, :35]
add_false_view1 = add_false_view1.ix[:, 1:]
add_label = np.array([0] * q + [1] * n).reshape((q + n, 1))
train_feat_view1 = np.concatenate((train_feat_view1, add_true_view1.values, add_false_view1.values), axis=0)
label_add_view1 = np.concatenate((label, add_label), axis=0)
U_ = U_.ix[q:U_array_add.shape[0] - n:]  # 剔除

x_train, x_val, y_train, y_val = train_test_split(train_feat_view1, label_add_view1, test_size=0.1, random_state=518)
clf_view1_1 = xgb.XGBClassifier(max_depth=5,
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
clf_view1_1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
##########################################在最初视图一的10万条数据上做预测，评价xgb性能########################

y_submission = clf_view1_1.predict_proba(test_feat_view1_original.values)[:, 0]
pred_original = {'id': test_id_original, 'prob': y_submission}
pred_original = pd.DataFrame.from_dict(pred_original)
pred_original = pd.concat([pred_original, test_feat_original], axis=1)
pred_original.sort_values(by='prob', ascending=[0], inplace=True)
pred_original = pred_original.reset_index(drop=True)
pred1 = pred_original.iloc[:20000]
right = len(pred1[pred1.id.isin(df_9982.id.unique())])
print('round %d，clf_view1_1  ' % (1))
print(' right number:{},score:{}'.format(right, right * 100.00 / 20000))

T = 5
flag = 1
#######round 2 开始 循环
for i in range(2, T + 2):
    flag = flag + 1
    num = 2 * (q + n)
    U_fill = U.sample(n=num, random_state=123, axis=0)
    U_id = U_fill.iloc[:, :1]
    id = np.array(U_id)
    # U 剔除 U'
    for i in range(0, 100000):
        if i in id:
            U = U.drop(i, axis=0)
        else:
            continue

    U_added = np.concatenate((U_, U_fill), axis=0)

    U_ = pd.DataFrame.from_dict(U_added)

    #############缓冲池视图划分###########
    U_view1 = U_.iloc[:, :36]
    U_view2 = U_.iloc[:, 36:]
    U_view1_original = U_view1
    U_view2_original = U_view2
    U_view1_test = U_view1.ix[:, 1:]
    U_view2_test = U_view2

    ##############################训练的学习器，对缓冲池中view1部分做预测,并按置信度排序###############
    y_submission = clf_view1_1.predict_proba(U_view1_test.values)[:, 0]

    test_id_view2 = test_id[:y_submission.shape[0]]

    pred_U_view1 = {'id': test_id_view2, 'prob': y_submission}
    pred_U_view1 = pd.DataFrame.from_dict(pred_U_view1)

    pred_U_view1 = pd.concat([pred_U_view1, U_view1_test], axis=1)
    pred_U_view1.sort_values(by='prob', ascending=[0], inplace=True)
    pred_U_view1 = pred_U_view1.reset_index(drop=True)

    pred_U_view1_id = pred_U_view1.ix[:, 0]
    id = np.array(pred_U_view1_id)
    U_array = np.array(U_)
    U_array_add = U_array[id - 1]  ######view2
    U_array_add = pd.DataFrame.from_dict(U_array_add)
    # U_view2_feat = U_view2_test_array_add

    add_true_view2 = U_array_add.ix[0:q - 1, 36:]
    add_false_view2 = U_array_add.ix[U_array_add.shape[0] - n:, 36:]
    add_label = np.array([0] * q + [1] * n).reshape((q + n, 1))
    train_feat_view2 = np.concatenate((train_feat_view2, add_true_view2.values, add_false_view2.values), axis=0)
    label_add_view2 = np.concatenate((label_add_view2, add_label), axis=0)
    U_ = U_.ix[q:U_array_add.shape[0] - n:]  # 剔除
    x_train, x_val, y_train, y_val = train_test_split(train_feat_view2, label_add_view2, test_size=0.1,
                                                      random_state=518)
    clf_view2_1 = xgb.XGBClassifier(max_depth=5,
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
    clf_view2_1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
    ##########################################在最初视图二的10万条数据上做预测，评价xgb性能￥########################

    y_submission = clf_view2_1.predict_proba(test_feat_view2_original.values)[:, 0]
    pred_original = {'id': test_id_original, 'prob': y_submission}
    pred_original = pd.DataFrame.from_dict(pred_original)
    pred_original = pd.concat([pred_original, test_feat_original], axis=1)
    pred_original.sort_values(by='prob', ascending=[0], inplace=True)
    pred_original = pred_original.reset_index(drop=True)
    pred1 = pred_original.iloc[:20000]
    right = len(pred1[pred1.id.isin(df_9982.id.unique())])

    print('round %d，clf2  ' % (flag))
    print(' right number:{},score:{}'.format(right, right * 100.00 / 20000))

    ##############################训练的学习器，对缓冲池中view2部分做预测,并按置信度排序###############
    U_view2_test = U_.iloc[:, 36:]
    U_view2_test = U_view2_test.reset_index(drop=True)
    y_submission = clf_view2_1.predict_proba(U_view2_test.values)[:, 0]
    test_id_view2 = test_id[:y_submission.shape[0]]
    pred_U_view2 = {'id': test_id_view2, 'prob': y_submission}
    pred_U_view2 = pd.DataFrame.from_dict(pred_U_view2)

    pred_U_view2 = pd.concat([pred_U_view2, U_view2_test], axis=1)
    pred_U_view2.sort_values(by='prob', ascending=[0], inplace=True)
    pred_U_view2 = pred_U_view2.reset_index(drop=True)

    pred_U_view2_id = pred_U_view2.ix[:, 0]
    id = np.array(pred_U_view2_id)
    U_array = np.array(U_)
    U_array_add = U_array[id - 1]  ######view2
    U_array_add = pd.DataFrame.from_dict(U_array_add)
    # U_view2_feat = U_view2_test_array_add

    add_true_view1 = U_array_add.ix[0:q - 1, :35]
    add_true_view1 = add_true_view1.ix[:, 1:]
    add_false_view1 = U_array_add.ix[U_array_add.shape[0] - n:, :35]
    add_false_view1 = add_false_view1.ix[:, 1:]
    add_label = np.array([0] * q + [1] * n).reshape((q + n, 1))
    train_feat_view1 = np.concatenate((train_feat_view1, add_true_view1.values, add_false_view1.values), axis=0)
    label_add_view1 = np.concatenate((label_add_view1, add_label), axis=0)
    U_ = U_.ix[q:U_array_add.shape[0] - n:]  # 剔除

    x_train, x_val, y_train, y_val = train_test_split(train_feat_view1, label_add_view1, test_size=0.1,
                                                      random_state=518)
    clf_view1_1 = xgb.XGBClassifier(max_depth=5,
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
    clf_view1_1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss')
    ##########################################在最初视图一的10万条数据上做预测，评价xgb性能########################

    y_submission = clf_view1_1.predict_proba(test_feat_view1_original.values)[:, 0]
    pred_original = {'id': test_id_original, 'prob': y_submission}
    pred_original = pd.DataFrame.from_dict(pred_original)
    pred_original = pd.concat([pred_original, test_feat_original], axis=1)
    pred_original.sort_values(by='prob', ascending=[0], inplace=True)
    pred_original = pred_original.reset_index(drop=True)
    pred1 = pred_original.iloc[:20000]
    right = len(pred1[pred1.id.isin(df_9982.id.unique())])
    print('round %d，clf1  ' % (flag))
    print(' right number:{},score:{}'.format(right, right * 100.00 / 20000))