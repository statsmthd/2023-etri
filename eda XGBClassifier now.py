#!/usr/bin/env python
# coding: utf-8

# In[37]:


import glob
import os
import pandas as pd
import numpy as np
import openpyxl


# # EDA 데이터 불러오기

# In[38]:


path = 'E:/연구과제/03. 2023 RIS/00. 공모전/공모전/KEMDy19/EDA'    #데이터폴더 경로 지정해주기 
parent_folders = os.listdir(path)                     #폴더 불러오기
path_li = []
for parent_folder in parent_folders:
    child_folders = []
    child_folders.append((np.array(os.listdir(path+"/"+parent_folder)).flatten().tolist())) # eda에 있는 폴더 다 불러오기
    temp_folders = (np.array(child_folders)).flatten().tolist()
    for folder in temp_folders:
        files = os.listdir(path+"/"+parent_folder+"/"+folder) #sessio01폴더 불러오기
        for file in files:
            if (path+"/"+parent_folder+folder+file).endswith('.csv') == True: # session01_impro01안에있는 csv 파일만 순서대로 path_li로 보내기
                path_li.append(path+"/"+parent_folder+"/"+folder+"/"+file)


# In[39]:


path_li_M = path_li[1::2]       # 모든 세션의 M.csv 파일 경로
path_li_F = path_li[0::2]       # 모든 세션의 F.csv 파일 경로


# In[40]:


total_eda_M=pd.DataFrame()
for filename_M in path_li_M:
    temp_label_M_df = pd.read_csv(filename_M, names = ['EDA', 'Period', 'Time', 'Segment ID'])
    total_eda_M = pd.concat([total_eda_M, temp_label_M_df], axis=0)
    total_eda_M = total_eda_M.dropna(subset = ['Segment ID'])       # Segment ID 결측값 삭제
total_eda_M     # 모든 세션 M.csv의 대화 바이오 데이터


# In[41]:


total_eda_F=pd.DataFrame()
for filename_F in path_li_F:
    temp_label_F_df = pd.read_csv(filename_F, names = ['EDA', 'Period', 'Time', 'Segment ID'])
    total_eda_F = pd.concat([total_eda_F, temp_label_F_df], axis=0)
    total_eda_F = total_eda_F.dropna(subset = ['Segment ID'])       # Segment ID 결측값 삭제
total_eda_F     # 모든 세션 F.csv의 대화 바이오 데이터


# EDA MF 데이터 불러오기

# In[42]:


total_eda_M_M = total_eda_M[total_eda_M['Segment ID'].str.contains('M0')]   # 모든 세션 M.csv 에서 M이 말할 때의 바이오 데이터
total_eda_F_F = total_eda_F[total_eda_F['Segment ID'].str.contains('F0')]   # 모든 세션 F.csv 에서 F가 말할 때의 바이오 데이터


# In[43]:


total_eda_MF = pd.concat([total_eda_M_M,total_eda_F_F])


# # 감정평가 MF 데이터 불러오기

# In[44]:


label_path = 'E:/연구과제/03. 2023 RIS/00. 공모전/공모전/KEMDy19_수정/Session_MF_uniq_csv'  # 감정레이블 경로설정
label_filename = os.listdir(label_path)                                  # 경로 설정한 폴더 열기
label_li = [label_file for label_file in label_filename if label_file.endswith('.csv')]  # .csv 파일만 불러오기


# In[45]:


total_label_df = pd.DataFrame()  
for filename in label_li:
    temp_label_df = pd.read_csv(label_path+'/'+filename)  # 감정레이블 파일 내용 읽어들이기
    total_label_df = pd.concat([total_label_df, temp_label_df], axis=0) 

total_label_df = total_label_df.reset_index(drop=True) 
print(len(total_label_df))
print(total_label_df.duplicated(['Segment ID']).value_counts())  # 중복값 확인하여 값 카운트

total_label_df = total_label_df[['Numb', 'Numb_re', 'Segment ID','Total Evaluation Emotion','Total Evaluation Valence','Total Evaluation Arousal']]
total_label_df


# # 데이터 feature 변환

# In[46]:


eda_value = total_eda_MF[['Segment ID','EDA']]

# Segment ID를 기준으로 groupby하고 ECG값을 리스트로 묶음
grouped = eda_value.groupby('Segment ID')['EDA'].agg(list)

# 최종 EDA label 데이터
eda_label_df = pd.merge(total_label_df, grouped, on='Segment ID', how='inner')


# In[47]:


# 가장 긴 EDA 신호의 길이 계산
max_len = max([len(x) for x in eda_label_df['EDA']])

# ECG 신호를 feature로 변환
X = []
for i in range(len(eda_label_df['EDA'])):
    eda_signal = eda_label_df['EDA'][i]
    eda_len = len(eda_signal)
    feature = eda_signal + [0] * (max_len - eda_len)
    X.append(feature)

# 리스트를 데이터프레임으로 변환
X = np.array(X)
eda_vector = pd.DataFrame(X)
eda_vector = eda_vector.iloc[:,:86]

# 감정 데이터와 EDA 결합
new_train = pd.concat([eda_label_df, eda_vector], axis=1)
new_train.drop(['EDA'], axis=1, inplace=True)


# # 데이터 split

# In[48]:


test_df = new_train['Segment ID'].str.contains('Sess03|Sess04|Sess10|Sess17')
train_df = new_train[~test_df]

test_df = new_train[new_train['Segment ID'].str.contains('Sess03|Sess04|Sess10|Sess17')]


# In[86]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[87]:


x_df = train_df.drop(['Numb', 'Numb_re', 'Segment ID','Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal'], axis=1)
y_df = train_df[['Total Evaluation Emotion']]

y_df = le.fit_transform(y_df)


# In[88]:


X_test = test_df.drop(['Numb', 'Numb_re', 'Segment ID','Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal'], axis=1)
y_test = test_df[['Total Evaluation Emotion']]

y_test = le.fit_transform(y_test)


# In[89]:


print(x_df.shape)
print(y_df.shape)
print(X_test.shape)
print(y_test.shape)


# In[52]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score, mean_absolute_percentage_error, f1_score
# from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pickle


# In[53]:


# 베이지안 최적화를 위한 탐색 공간 설정
search_spaces = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'min_child_weight': Integer(1, 10),
    'max_depth': Integer(3, 15),
    'subsample': Real(0.5, 1.0, 'uniform'),
    'colsample_bytree': Real(0.1, 1.0, 'uniform'),
    'reg_alpha': Real(1e-5, 1.0, 'log-uniform'),
    'reg_lambda': Real(1e-5, 1000, 'log-uniform'),
    'gamma': Real(1e-5, 0.5, 'log-uniform'),
    'n_estimators': Integer(50, 500),
    'booster': Categorical(['gbtree', 'gblinear', 'dart'])
}


# In[54]:


# XGBClassifier 모델 정의
xgb_class = XGBClassifier()

# 베이지안 최적화 + 교차검증
clf = BayesSearchCV(
    xgb_class,
    search_spaces,
    cv=10,  # 교차검증 폴드 수
    n_iter=32,  # 최대 탐색 횟수
    scoring='accuracy',  # 목적함수 설정
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    random_state=42
)


# In[55]:


# 모델 훈련
clf.fit(x_df, y_df)


# In[25]:


# Print the best hyperparameters and accuracy score 모델 훈련 정확도
print("Best hyperparameters: ", clf.best_params_)
print("Accuracy score: ", clf.best_score_)


# In[57]:


preds = clf.best_estimator_.predict(X_test)
f1 = f1_score(y_test, preds, average='micro')
accuracy = accuracy_score(y_test, preds)
print('현감정 F1 스코어 : {0:.4f}'.format(f1))
print('현감정 test Accuracy : {0:.4f}'.format(accuracy))


# In[58]:


# Save the trained model to a file using pickle 모델 훈련시킨거 저장하기

filename = 'xgb_classifier_model_now.pkl'
pickle.dump(clf, open(filename, 'wb'))


# Load the saved model from the file and use it to make predictions on the test set 훈련모델 저장한거 불러오기

# loaded_model = pickle.load(open(filename, 'rb'))
# y_pred = loaded_model.predict(X_test)


# In[90]:


# 예측데이터 저장

nn = pd.DataFrame(preds, columns=['pred'])
nn.to_csv('XGBClassifier_now.csv', index=False)

