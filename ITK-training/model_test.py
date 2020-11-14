#!/usr/bin/env python
# coding: utf-8

# # 强弱模型测试

# In[2]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[3]:


train_feat = pd.read_csv('train.csv', header=None)
test_feat = pd.read_csv('testA.csv', header=None)
label = pd.read_csv('label.txt', header=None)
###处理填补inf，NaN值
train_feat = train_feat.replace([np.inf, -np.inf], np.nan)
train_feat = train_feat.fillna(0) #替换正负inf为NA
test_feat = test_feat.replace([np.inf, -np.inf], np.nan).fillna(0) #替换正负inf为NA
#compare to s9982
df_9982 = pd.read_csv("s9982.txt",header=None)
df_9982.columns = ['id']
print(len(df_9982))

train_feat = train_feat.ix[:, 1:]
test_id = test_feat.ix[:, 0]
test_feat = test_feat.ix[:, 1:]

train_feat = train_feat.ix[:, 1:]
test_id = test_feat.ix[:, 0]
test_feat = test_feat.ix[:, 1:]
test_feat_local = test_feat.iloc[:,35:]
train_feat_local = train_feat.iloc[:,35:]
test_feat_local_original = test_feat_local


# In[7]:


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
#train_feat = train_feat_local
#test_feat = test_feat_local



# In[8]:


###处理填补inf，NaN值
train_feat = train_feat.replace([np.inf, -np.inf], np.nan)
train_feat = train_feat.fillna(0) #替换正负inf为NA
test_feat = test_feat.replace([np.inf, -np.inf], np.nan).fillna(0) #替换正负inf为NA
x_train, x_val, y_train, y_val = train_test_split(train_feat, label, test_size=0.05, random_state=518)


# In[9]:


train_feat


# In[10]:


test_feat


# In[11]:


print(x_train.isnull().any())


# # 模型测试-XGB

# In[12]:


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
clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='logloss',verbose = False)
y_submission = clf.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])


wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('XGB right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# In[14]:


y_submission.shape


# # 模型测试-LR

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[8]:


#x_train = x_train.reset_index(drop=True)
#y_train = y_train.reset_index(drop=True)


# In[9]:


LR = LogisticRegression()
LR.fit(train_feat,label)


# In[10]:


y_submission = LR.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])


wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('LR right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# In[1]:


y_submission


# # 模型测试-决策树

# In[21]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=30,class_weight='balanced')

clf.fit(train_feat,label) #此时完成训练


# In[22]:


y_submission = clf.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('DT right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# # 模型测试-RF

# In[23]:


from sklearn.ensemble import RandomForestClassifier    #导入需要的模块
rfc = RandomForestClassifier()  #实例化     

rfc = rfc.fit(x_train,y_train) #用训练集数据训练

y_submission = rfc.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('RF right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# # 模型测试-GBDT

# In[54]:


train_feat


# In[63]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(train_feat,label)


# In[64]:


y_submission = clf.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('RF right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# # 模型测试-ANN

# In[83]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# In[84]:


ss = StandardScaler()
x_train = pd.DataFrame(ss.fit_transform(x_train))
test_feat = pd.DataFrame(ss.fit_transform(test_feat))


# In[85]:


ann = MLPClassifier(solver='lbfgs', alpha=0.000000001, hidden_layer_sizes=(20,), random_state=1)
ann.fit(x_train, y_train)
# 模型效果获取
r = ann.score(x_train, y_train)
print("R值(准确率):", r)


# In[86]:


y_submission = ann.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('RF right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# # 模型测试-SVM

# In[87]:


from sklearn.svm import SVC  
SVM = SVC(kernel='rbf', probability=True)  
SVM.fit(x_train, y_train)  


# In[88]:


y_submission = SVM.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('SVM right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# In[89]:



from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC  
model = SVC(kernel='rbf', probability=True)  
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
grid_search.fit(x_train, y_train) 
best_parameters = grid_search.best_estimator_.get_params()  
for para, val in list(best_parameters.items()):  
    print(para, val)  
model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
model.fit(x_train, y_train)  


# In[90]:


y_submission = model.predict_proba(test_feat)[:, 0]
pred = {'id': test_id, 'prob': y_submission}
pred = pd.DataFrame.from_dict(pred)
pred = pd.concat([pred, test_feat], axis=1)
pred.sort_values(by='prob', ascending=[0], inplace=True)
pred = pred.reset_index(drop=True)

pred1 = pred.iloc[:20000]
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
right=len(pred1[pred1.id.isin(df_9982.id.unique())])
wrong = 20000 - right
P = right/20000
R = right/20000
F = 5*P*R*100.00/(2*P+3*R)
print('SVM right_num:{},wrong_num:{},P:{},R:{},F:{}'.format(right,wrong,P,R,F))


# In[ ]:




