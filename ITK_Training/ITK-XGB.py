#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn
import warnings
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import  pyplot
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt

label_range = 20000
print(label_range)
train_feat = pd.read_csv('train.csv', header=None)
test_feat = pd.read_csv('testA.csv', header=None)
label = pd.read_csv('label.txt', header=None)
new_test_feat = pd.read_csv('new_test_feat.csv')

#compare to s9982
df_9982 = pd.read_csv("s9982.txt",header=None)
df_9982.columns = ['id']
print(len(df_9982))

train_feat = train_feat.ix[:, 1:]
test_id = test_feat.ix[:, 0]
test_feat = test_feat.ix[:, 1:]
test_feat_local = test_feat.iloc[:,35:]
train_feat_local = train_feat.iloc[:,35:]
test_feat_local_original = test_feat_local

train_feat_local_fold_1 = train_feat_local.ix[:1000]
train_feat_local_fold_2 = train_feat_local.ix[1001:2000]
train_feat_local_fold_3 = train_feat_local.ix[2001:3000]
label_fold_1 = label.ix[:1000]
label_fold_2 = label.ix[1001:2000]
label_fold_3 = label.ix[2001:3000]
train_feat_local_fold_1_2 = pd.concat([train_feat_local_fold_1, train_feat_local_fold_2], axis=0)
train_feat_local_fold_1_3 = pd.concat([train_feat_local_fold_1, train_feat_local_fold_3], axis=0)
train_feat_local_fold_2_3 = pd.concat([train_feat_local_fold_2, train_feat_local_fold_3], axis=0)
label_fold_1_2 = pd.concat([label_fold_1, label_fold_2], axis=0)
label_fold_1_3 = pd.concat([label_fold_1, label_fold_3], axis=0)
label_fold_2_3 = pd.concat([label_fold_2, label_fold_3], axis=0)


# In[ ]:



x_train, x_val, y_train, y_val = train_test_split(train_feat_local_fold_1_2, label_fold_1_2, test_size=0.05, random_state=518)
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
                        scale_pos_weight=1,
                        silent=1
                        )
clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)
############################################################
y_submission1 = clf1.predict_proba(train_feat_local_fold_3)[:, 0]
prob_test = pd.DataFrame.from_dict(y_submission1)
prob_test.columns = ['71']
prob_test.index  = train_feat_local_fold_3.index
new_train_feat_fold_3 = pd.concat([train_feat_local_fold_3, prob_test], axis=1)

x_train, x_val, y_train, y_val = train_test_split(train_feat_local_fold_1_3, label_fold_1_3, test_size=0.05, random_state=518)
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
                        scale_pos_weight=1,
                        silent=1
                        )
clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)
############################################################
y_submission1 = clf1.predict_proba(train_feat_local_fold_2)[:, 0]
prob_test = pd.DataFrame.from_dict(y_submission1)
prob_test.columns = ['71']
prob_test.index  = train_feat_local_fold_2.index
new_train_feat_fold_2 = pd.concat([train_feat_local_fold_2, prob_test], axis=1)

x_train, x_val, y_train, y_val = train_test_split(train_feat_local_fold_2_3, label_fold_2_3, test_size=0.05, random_state=518)
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
                        scale_pos_weight=1,
                        silent=1
                        )
clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)
############################################################
y_submission1 = clf1.predict_proba(train_feat_local_fold_1)[:, 0]
prob_test = pd.DataFrame.from_dict(y_submission1)
prob_test.columns = ['71']
prob_test.index  = train_feat_local_fold_1.index
new_train_feat_fold_1 = pd.concat([train_feat_local_fold_1, prob_test], axis=1)

new_train_feat_36 = pd.concat([new_train_feat_fold_1, new_train_feat_fold_2,new_train_feat_fold_3], axis=0)




new_test_feat = pd.read_csv('new_test_feat.csv')

new_train_feat = pd.read_csv('new_train_feat.csv')
new_train_feat = new_train_feat.ix[:,1:]
new_test_feat = new_test_feat.ix[:,1:]
new_test_feat 

train_feat_cloud =  pd.concat([new_train_feat.ix[:,:35],new_train_feat_36], axis=1)
train_feat_cloud



fig, ax = plt.subplots(figsize=(12,12))

xgb.plot_importance(clf1, height=0.8, ax=ax)
plt.show()

y_submission1 = pd.DataFrame.from_dict(y_submission1)
y_submission1.columns = ['prob']

y_submission1.sort_values(by='prob', ascending=[0], inplace=True)

prob_test= y_submission1.reset_index(drop=True)
y_true = [1]*20000 + [0]*80000
y_true = pd.DataFrame.from_dict(y_true)

y_true.columns = ['true label']
data = pd.concat([prob_test, y_true], axis=1)


x_train, x_val, y_train, y_val = train_test_split(train_feat_local, label, test_size=0.05, random_state=518)
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
clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)
##########################################################
y_submission1 = clf1.predict_proba(test_feat_local)[:, 0]
pred_view1 = {'id': test_id, 'prob': y_submission1}
pred_view1 = pd.DataFrame.from_dict(pred_view1)
pred_view1 = pd.concat([pred_view1, test_feat_local], axis=1)
pred_view1.sort_values(by='prob', ascending=[0], inplace=True)
pred_view1 = pred_view1.reset_index(drop=True)
pred1 = pred_view1.iloc[:label_range]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
print('local')
wrong = label_range - right
P = right/label_range
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('local right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))







new_train_feat.shape



x_train, x_val, y_train, y_val = train_test_split(train_feat_cloud, label, test_size=0.05, random_state=518)
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
                        scale_pos_weight=1,
                        silent=1
                        )
clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)

y_submission1 = clf1.predict_proba(new_test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission1}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, new_test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)
pred1 = pred.iloc[:label_range]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = label_range - right
P = right/label_range
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)

print('cloud' )
print('right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


pred

y_true = [1]*20000+[0]*80000
y_true = pd.DataFrame.from_dict(y_true)
y_true.shape



fig, ax = plt.subplots(figsize=(12,12))

xgb.plot_importance(clf1, height=0.8, ax=ax)
plt.show()

y_submission1 = pd.DataFrame.from_dict(y_submission1)
y_submission1.columns = ['prob']

y_submission1.sort_values(by='prob', ascending=[0], inplace=True)

prob_test= y_submission1.reset_index(drop=True)
y_true 
y_true = pd.DataFrame.from_dict(y_true)

y_true.columns = ['true label']
data = pd.concat([prob_test, y_true], axis=1)



new_train_feat = train_feat_cloud



p = 12000
n = 36000
T = 4

for i in range(1,T):
    

   

    add_true = pred.ix[0:p-1, 2:]
    add_false = pred.ix[100000-n:, 2:]
    add_label = np.array([0] * p + [1] * n).reshape((p+n, 1))
    new_train_feat_add = np.concatenate((new_train_feat, add_true.values, add_false.values), axis=0)
    
    label_add = np.concatenate((label, add_label), axis=0)
    print("Data Size")
    print(label_add.shape[0])
    x_train, x_val, y_train, y_val = train_test_split(new_train_feat_add, label_add, test_size=0.1, random_state=518)
    idx = np.random.permutation(label.size)
    new_train_feat_add = new_train_feat_add[idx]
    label_add = label_add[idx]

    clf1 = xgb.XGBClassifier(max_depth=5,
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
                            scale_pos_weight=1,
                            silent=1
                            )
    clf1.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)
    y_submission1 = clf1.predict_proba(new_test_feat.values)[:, 0]

    pred = {'id': test_id, 'prob': y_submission1}
    pred = pd.DataFrame.from_dict(pred)
    pred = pd.concat([pred, new_test_feat], axis=1)
    pred.sort_values(by='prob', ascending=[0], inplace=True)
    pred = pred.reset_index(drop=True)
    pred1 = pred.iloc[:label_range]

    right=len(pred1[pred1.id.isin(df_9982.id.unique())])
    wrong = label_range - right
    P = right/label_range
    R = right/20000
    F = 5*P*R*100.00/(2*P+3*R)

    print('round %d' %(i))
    print('ITK right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))

    y_submission1 = pd.DataFrame.from_dict(y_submission1)
    y_submission1.columns = ['prob']

    y_submission1.sort_values(by='prob', ascending=[0], inplace=True)

    prob_test= y_submission1.reset_index(drop=True)
    y_true 
    y_true = pd.DataFrame.from_dict(y_true)

    y_true.columns = ['true label']
    data = pd.concat([prob_test, y_true], axis=1)
  
    p = int(1.3*p)
    n = int(1.3*n)
    print("1.2")

