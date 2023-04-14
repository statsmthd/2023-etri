#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import os
import pandas as pd
import numpy as np
import openpyxl


# # EDA 데이터 불러오기

# In[3]:


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


# In[4]:


path_li_M = path_li[1::2]       # 모든 세션의 M.csv 파일 경로
path_li_F = path_li[0::2]       # 모든 세션의 F.csv 파일 경로


# In[5]:


total_eda_M=pd.DataFrame()
for filename_M in path_li_M:
    temp_label_M_df = pd.read_csv(filename_M, names = ['EDA', 'Period', 'Time', 'Segment ID'])
    total_eda_M = pd.concat([total_eda_M, temp_label_M_df], axis=0)
    total_eda_M = total_eda_M.dropna(subset = ['Segment ID'])       # Segment ID 결측값 삭제
total_eda_M     # 모든 세션 M.csv의 대화 바이오 데이터


# In[6]:


total_eda_F=pd.DataFrame()
for filename_F in path_li_F:
    temp_label_F_df = pd.read_csv(filename_F, names = ['EDA', 'Period', 'Time', 'Segment ID'])
    total_eda_F = pd.concat([total_eda_F, temp_label_F_df], axis=0)
    total_eda_F = total_eda_F.dropna(subset = ['Segment ID'])       # Segment ID 결측값 삭제
total_eda_F     # 모든 세션 F.csv의 대화 바이오 데이터


# EDA MF 데이터 불러오기

# In[7]:


total_eda_M_M = total_eda_M[total_eda_M['Segment ID'].str.contains('M0')]   # 모든 세션 M.csv 에서 M이 말할 때의 바이오 데이터
total_eda_F_F = total_eda_F[total_eda_F['Segment ID'].str.contains('F0')]   # 모든 세션 F.csv 에서 F가 말할 때의 바이오 데이터


# In[8]:


total_eda_MF = pd.concat([total_eda_M_M,total_eda_F_F])


# # 감정평가 MF 데이터 불러오기

# In[9]:


label_path = 'E:/연구과제/03. 2023 RIS/00. 공모전/공모전/KEMDy19_수정/Session_MF_uniq_csv'  # 감정레이블 경로설정
label_filename = os.listdir(label_path)                                  # 경로 설정한 폴더 열기
label_li = [label_file for label_file in label_filename if label_file.endswith('.csv')]  # .csv 파일만 불러오기


# In[10]:


total_label_df = pd.DataFrame()  
for filename in label_li:
    temp_label_df = pd.read_csv(label_path+'/'+filename)  # 감정레이블 파일 내용 읽어들이기
    total_label_df = pd.concat([total_label_df, temp_label_df], axis=0) 

total_label_df = total_label_df.reset_index(drop=True) 
print(len(total_label_df))
print(total_label_df.duplicated(['Segment ID']).value_counts())  # 중복값 확인하여 값 카운트

total_label_df = total_label_df[['Numb', 'Numb_re', 'Segment ID','Total Evaluation Emotion','Total Evaluation Valence','Total Evaluation Arousal']]
total_label_df


# # 데이터 벡터화

# 남자 데이터

# In[22]:


eda_value_M = total_eda_M[['Segment ID','EDA']]

# Segment ID를 기준으로 groupby하고 EDA값을 리스트로 묶음
grouped_M = eda_value_M.groupby('Segment ID')['EDA'].agg(list)

# 최종 EDA label 데이터
eda_M_label_df = pd.merge(total_label_df, grouped_M, on='Segment ID', how='inner')

# 가장 긴 EDA 신호의 길이 계산
max_len = max([len(x) for x in eda_M_label_df['EDA']])

# EDA 신호를 feature로 변환
X = []
for i in range(len(eda_M_label_df['EDA'])):
    eda_signal = eda_M_label_df['EDA'][i]
    eda_len = len(eda_signal)
    feature = eda_signal + [0] * (max_len - eda_len)
    X.append(feature)

# 리스트를 데이터프레임으로 변환
X = np.array(X)
eda_M_vector = pd.DataFrame(X)
eda_M_vector = eda_M_vector.iloc[:,:86]

# 감정 데이터와 EDA 결합
new_train_M = pd.concat([eda_M_label_df, eda_M_vector], axis=1)
new_train_M.drop(['EDA'], axis=1, inplace=True)


# In[23]:


shift_eda_M = eda_M_vector.shift(1)
eda_M_vector.columns = [100 + num for num, col in enumerate(eda_M_vector.columns)]


# In[24]:


result_bef_M_df = pd.concat([new_train_M[['Numb', 'Numb_re', 'Segment ID', 'Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal']], shift_eda_M], axis=1)
result_aft_M_df = pd.concat([new_train_M[['Numb', 'Numb_re', 'Segment ID', 'Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal']], eda_M_vector], axis=1)


# In[25]:


col_list = ['Sess01_script01_M001', 'Sess01_script02_F001', 'Sess01_script03_F001', 'Sess01_script04_M001', 'Sess01_script05_M001', 'Sess01_script06_F001', 'Sess01_impro01_F001', 'Sess01_impro02_M001', 'Sess01_impro03_M001', 'Sess01_impro04_M001',
'Sess02_script01_F001', 'Sess02_script02_M001', 'Sess02_script03_F001', 'Sess02_script04_M001', 'Sess02_script05_M001', 'Sess02_script06_F001', 'Sess02_impro01_F001', 'Sess02_impro02_F001', 'Sess02_impro03_M001', 'Sess02_impro04_F001',
'Sess03_script01_M001', 'Sess03_script02_F001', 'Sess03_script03_M001', 'Sess03_script04_F001', 'Sess03_script05_M001', 'Sess03_script06_F001', 'Sess03_impro01_F001', 'Sess03_impro02_F001', 'Sess03_impro03_F001', 'Sess03_impro04_M001',
'Sess04_script01_F001', 'Sess04_script02_F001', 'Sess04_script03_M001', 'Sess04_script04_M001', 'Sess04_script05_M001', 'Sess04_script06_F001', 'Sess04_impro01_F001', 'Sess04_impro02_M001', 'Sess04_impro03_M001', 'Sess04_impro04_M001',
'Sess05_script01_F001', 'Sess05_script02_F001', 'Sess05_script03_F001', 'Sess05_script04_M001', 'Sess05_script05_M001', 'Sess05_script06_M001', 'Sess05_impro01_F001', 'Sess05_impro02_M001', 'Sess05_impro03_F001', 'Sess05_impro04_M001',
'Sess06_script01_F001', 'Sess06_script02_F001', 'Sess06_script03_M001', 'Sess06_script04_M001', 'Sess06_script05_F001', 'Sess06_script06_M001', 'Sess06_impro01_M001', 'Sess06_impro02_M001', 'Sess06_impro03_M001', 'Sess06_impro04_F001',
'Sess07_script01_F001', 'Sess07_script02_M001', 'Sess07_script03_F001', 'Sess07_script04_F001', 'Sess07_script05_M001', 'Sess07_script06_M001', 'Sess07_impro01_M001', 'Sess07_impro02_F001', 'Sess07_impro03_F001', 'Sess07_impro04_M001',
'Sess08_script01_M001', 'Sess08_script02_M001', 'Sess08_script03_M001', 'Sess08_script04_F001', 'Sess08_script05_F001', 'Sess08_script06_F001', 'Sess08_impro01_F001', 'Sess08_impro02_M001', 'Sess08_impro03_F001', 'Sess08_impro04_M001',
'Sess09_script01_F001', 'Sess09_script02_M001', 'Sess09_script03_M001', 'Sess09_script04_M001', 'Sess09_script05_F001', 'Sess09_script06_F001', 'Sess09_impro01_F001', 'Sess09_impro02_M001', 'Sess09_impro03_M001', 'Sess09_impro04_F001',
'Sess10_script01_M001', 'Sess10_script02_F001', 'Sess10_script03_M001', 'Sess10_script04_F001', 'Sess10_script05_M001', 'Sess10_script06_F001', 'Sess10_impro01_F001', 'Sess10_impro02_M001', 'Sess10_impro03_F001', 'Sess10_impro04_M001',
'Sess11_script01_F001', 'Sess11_script02_F001', 'Sess11_script03_M001', 'Sess11_script04_F001', 'Sess11_script05_M001',' Sess11_script06_M001', 'Sess11_impro01_F001', 'Sess11_impro02_F001', 'Sess11_impro03_F001', 'Sess11_impro04_M001',
'Sess12_script01_M001', 'Sess12_script02_F001', 'Sess12_script03_M001', 'Sess12_script04_M001', 'Sess12_script05_F001', 'Sess12_script06_F001', 'Sess12_impro01_M001', 'Sess12_impro02_F001', 'Sess12_impro03_F001', 'Sess12_impro04_M001',
'Sess13_script01_F001', 'Sess13_script02_M001', 'Sess13_script03_F001', 'Sess13_script04_M001', 'Sess13_script05_M001', 'Sess13_script06_F001', 'Sess13_impro01_F001', 'Sess13_impro02_M001', 'Sess13_impro03_M001', 'Sess13_impro04_F001',
'Sess14_script01_F001', 'Sess14_script02_F001', 'Sess14_script03_M001', 'Sess14_script04_M001', 'Sess14_script05_M001', 'Sess14_script06_F001', 'Sess14_impro01_M001', 'Sess14_impro02_F001', 'Sess14_impro03_F001', 'Sess14_impro04_M001',
'Sess15_script01_F001', 'Sess15_script02_F001', 'Sess15_script03_M001', 'Sess15_script04_M001', 'Sess15_script05_M001', 'Sess15_script06_F001', 'Sess15_impro01_M001', 'Sess15_impro02_F001', 'Sess15_impro03_M001', 'Sess15_impro04_F001',
'Sess16_script01_F001', 'Sess16_script02_M001', 'Sess16_script03_F001', 'Sess16_script04_M001', 'Sess16_script05_M001', 'Sess16_script06_F001', 'Sess16_impro01_M001', 'Sess16_impro02_F001', 'Sess16_impro03_M001', 'Sess16_impro04_F001',
'Sess17_script01_F001', 'Sess17_script02_M001', 'Sess17_script03_M001', 'Sess17_script04_F001', 'Sess17_script05_M001', 'Sess17_script06_F001', 'Sess17_impro01_F001', 'Sess17_impro02_M001', 'Sess17_impro03_F001', 'Sess17_impro04_F001',
'Sess18_script01_F001', 'Sess18_script02_M001', 'Sess18_script03_F001', 'Sess18_script04_F001', 'Sess18_script05_M001', 'Sess18_script06_M001', 'Sess18_impro01_M001', 'Sess18_impro02_F001', 'Sess18_impro03_M001', 'Sess18_impro04_M001',
'Sess19_script01_M001', 'Sess19_script02_M001', 'Sess19_script03_F001', 'Sess19_script04_F001', 'Sess19_script05_M001', 'Sess19_script06_F001', 'Sess19_impro01_F001', 'Sess19_impro02_F001', 'Sess19_impro03_M001', 'Sess19_impro04_M001',
'Sess20_script01_M001', 'Sess20_script02_F001', 'Sess20_script03_F001', 'Sess20_script04_F001', 'Sess20_script05_M001', 'Sess20_script06_M001', 'Sess20_impro01_M001', 'Sess20_impro02_M001', 'Sess20_impro03_F001', 'Sess20_impro04_F001' ]


# In[26]:


result_bef_M_df[result_bef_M_df['Segment ID'].isin(col_list)] = 0.0  # 전감정 스크립트 첫번째 결측값으로 대체

bef_eda_M_df = result_bef_M_df.iloc[:,6:]    # 전감정
aft_eda_M_df = result_aft_M_df.iloc[:,6:]   # 현감정

eda_bf_M_label = new_train_M[['Numb', 'Numb_re', 'Segment ID', 'Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal']]


# In[27]:


total_result_M_df = pd.concat([eda_bf_M_label, bef_eda_M_df, aft_eda_M_df], axis=1)
total_result_M_df


# In[28]:


total_result_M_df[total_result_M_df['Segment ID']=='Sess12_script02_F001']


# 여자 데이터
# 

# In[64]:


eda_value_F= total_eda_F[['Segment ID','EDA']]

# Segment ID를 기준으로 groupby하고 EDA값을 리스트로 묶음
grouped_F = eda_value_F.groupby('Segment ID')['EDA'].agg(list)

# 최종 EDA label 데이터
eda_F_label_df = pd.merge(total_label_df, grouped_F, on='Segment ID', how='inner')

# 가장 긴 EDA 신호의 길이 계산
max_len = max([len(x) for x in eda_F_label_df['EDA']])

# EDA 신호를 feature로 변환
X = []
for i in range(len(eda_F_label_df['EDA'])):
    eda_signal = eda_F_label_df['EDA'][i]
    eda_len = len(eda_signal)
    feature = eda_signal + [0] * (max_len - eda_len)
    X.append(feature)

# 리스트를 데이터프레임으로 변환
X = np.array(X)
eda_F_vector = pd.DataFrame(X)
eda_F_vector = eda_F_vector.iloc[:,:86]

# 감정 데이터와 EDA 결합
new_train_F = pd.concat([eda_F_label_df, eda_F_vector], axis=1)
new_train_F.drop(['EDA'], axis=1, inplace=True)


# In[30]:


shift_eda_F = eda_F_vector.shift(1)
eda_F_vector.columns = [100 + num for num, col in enumerate(eda_F_vector.columns)]


# In[31]:


result_bef_F_df = pd.concat([new_train_F[['Numb', 'Numb_re', 'Segment ID', 'Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal']], shift_eda_F], axis=1)
result_aft_F_df = pd.concat([new_train_F[['Numb', 'Numb_re', 'Segment ID', 'Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal']], eda_F_vector], axis=1)


# In[32]:


result_bef_F_df[result_bef_F_df['Segment ID'].isin(col_list)] = 0  # 전감정 스크립트 첫번째 결측값으로 대체

bef_eda_F_df = result_bef_F_df.iloc[:,6:]    # 전감정
aft_eda_F_df = result_aft_F_df.iloc[:,6:]   # 현감정

eda_bf_F_label = new_train_F[['Numb', 'Numb_re', 'Segment ID', 'Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal']]


# In[33]:


total_result_F_df = pd.concat([eda_bf_F_label, bef_eda_F_df, aft_eda_F_df], axis=1)
total_result_F_df


# In[34]:


total_result_F_df[total_result_F_df['Segment ID']=='Sess12_script02_F001']


# 남자여자 말하는 데이터만 합치기

# In[35]:


bef_aft_M_df = total_result_M_df[total_result_M_df['Segment ID'].str.contains('M0')]   # 모든 세션 M.csv 에서 M이 말할 때의 바이오 데이터
bef_aft_F_df = total_result_F_df[total_result_F_df['Segment ID'].str.contains('F0')]   # 모든 세션 F.csv 에서 F가 말할 때의 바이오 데이터

bef_aft_M_df.drop(['Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal'], axis=1, inplace=True)
bef_aft_F_df.drop(['Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal'], axis=1, inplace=True)


# In[36]:


bef_aft_MF_df = pd.concat([bef_aft_M_df, bef_aft_F_df], axis=0)
bef_aft_MF_df


# In[37]:


total_bf_label_df = pd.merge(total_label_df, bef_aft_MF_df, how='right', on=['Numb', 'Numb_re', 'Segment ID'])
total_bf_label_df.drop_duplicates(['Numb', 'Numb_re', 'Segment ID'], keep='first', inplace=True)
total_bf_label_df


# # 데이터 split

# In[38]:


test_bf_df = total_bf_label_df['Segment ID'].str.contains('Sess03|Sess04|Sess10|Sess17')
train_bf_df = total_bf_label_df[~test_bf_df]

test_bf_df = total_bf_label_df[total_bf_label_df['Segment ID'].str.contains('Sess03|Sess04|Sess10|Sess17')]


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


# In[101]:


x_bf_df = train_bf_df.drop(['Numb', 'Numb_re', 'Segment ID','Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal'], axis=1)
y_bf_df = train_bf_df[['Total Evaluation Emotion']]

y_bf_df = le.fit_transform(y_bf_df)


# In[102]:


xt_df = test_bf_df.drop(['Numb', 'Numb_re', 'Segment ID','Total Evaluation Emotion', 'Total Evaluation Valence', 'Total Evaluation Arousal'], axis=1)
yt_df = test_bf_df[['Total Evaluation Emotion']]

y_test = le.fit_transform(yt_df)


# In[104]:


print(x_bf_df.shape)
print(y_bf_df.shape)
print(xt_df.shape)
print(y_test.shape)


# # xgb 모델

# In[ ]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score, mean_absolute_percentage_error, f1_score
# from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pickle


# In[43]:


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


# In[51]:


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


# In[52]:


# 모델 훈련
clf.fit(x_bf_df, y_bf_df)


# In[60]:


# Print the best hyperparameters and accuracy score 모델 훈련 정확도
print("Best hyperparameters: ", clf.best_params_)
print("Accuracy score: ", clf.best_score_)


# In[95]:


# y_pred = clf.best_estimator_.predict(xt_df)
f1 = f1_score(y_test, y_pred, average='micro')
accuracy = accuracy_score(y_test, y_pred)
print('전+현감정 F1 스코어 : {0:.4f}'.format(f1))
print('전+현감정 test Accuracy : {0:.4f}'.format(accuracy))


# In[93]:


# Save the trained model to a file using pickle 모델 훈련시킨거 저장하기

filename = 'xgb_classifier_model_bf.pkl'
pickle.dump(clf, open(filename, 'wb'))


# Load the saved model from the file and use it to make predictions on the test set 훈련모델 저장한거 불러오기

# loaded_model = pickle.load(open(filename, 'rb'))
# y_pred = loaded_model.predict(xt_df)


# In[96]:


# 예측데이터 저장

nn = pd.DataFrame(y_pred, columns=['pred'])
nn.to_csv('XGBClassifier_bf.csv', index=False)

