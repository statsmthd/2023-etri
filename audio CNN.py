#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from glob import glob
from scipy.io import wavfile
import librosa.display
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import load_model

import soundfile as sf
import librosa

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


# In[2]:


path = 'E:/Competition/ETRI Competition/RAW DATA/KEMDy19/wav/train'    #데이터폴더 경로 지정해주기 
parent_folders = os.listdir(path)                     #폴더 불러오기
path_li = []
total_sum = 0
for parent_folder in parent_folders:
    child_folders = []
    child_folders.append((np.array(os.listdir(path+"/"+parent_folder)).flatten().tolist())) # wav에 있는 폴더 다 불러오기
    temp_folders = (np.array(child_folders)).flatten().tolist()
    for folder in temp_folders:
        files = os.listdir(path+"/"+parent_folder+"/"+folder) #sessio01폴더 불러오기
        for file in files:
            if (path+"/"+parent_folder+'/'+folder+'/'+file).endswith('.wav') == True: # session01_impro01안에있는 txt 파일만 순서대로 path_li로 보내기
                path_li.append(path+"/"+parent_folder+"/"+folder+"/"+file)

path_li[:3]


# In[3]:


print(len(path_li))


# ## 14-45초 음성 파일을 10초자리로 sliding_window 진행

# In[4]:


# 14초~45초 되는 음성 파일들을 가져옴
audio_files = []
for filename in path_li: # 음성 파일이 저장된 폴더 경로
    y, sr = librosa.load(filename, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 14 or duration >= 45:
        continue
    audio_files.append(filename)


# In[ ]:


len(audio_files)


# In[ ]:


# sliding window 크기 (10초)
window_size = 10

# sliding window 간격 (10초)
hop_size = 5

# 14초~45초 되는 음성 파일들을 저장할 폴더 경로
sliding_window = 'E:/Competition/ETRI Competition/RAW DATA/KEMDy19/wav/train/sliding window'


# In[ ]:


# sliding window을 적용하여 데이터 쪼개기
for file_path in audio_files:
    filename = os.path.basename(file_path)
    y, sr = librosa.load(file_path, sr=16000, offset=0)
    for i in range(0, len(y) - window_size * sr + hop_size * sr, hop_size * sr):
        if i + window_size * sr > len(y):
            window = y[-window_size * sr:]
        else:
            window = y[i:i + window_size * sr]
        output_file = os.path.join(sliding_window, f'{filename}_{i+window_size}s.wav')
        sf.write(output_file, window, sr, subtype='PCM_16')


# In[ ]:


for file_delete in audio_files:  #14초이상 45초 미만 음성파일 삭제
    os.remove(file_delete)


# ## 음성 mfcc 추출

# In[3]:


# 음성 파일을 feaher로 생성

max_pad_len = 3300
hop_length = 160  # 전체 frame 수 (hop_length의 길이만큼 옆으로 가면서 데이터를 읽는 옵션, 10ms를 기본으로 하고 있어 16000Hz인 음성에서는 160에 해당)
n_fft = 400       # frame 하나당 sample 수(일반적으로 자연어 처리에서는 음성을 25m의 크기를 기본으로 하고 있으며 16000Hz인 음성에서는 400에 해당하는 값) 

out = []
for file in path_li:
    data, sr = librosa.load(file)  # librosa로 음성 데이터 불러오기
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=128)  #mfcc로 음성을 array형태로 나타냄
    # print(mfccs)
    pad_width = max_pad_len - mfccs.shape[1]  # shape 만들기
    if pad_width > 0:
        mfcc_sh = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        out.append(mfcc_sh)
    else:
        out.append(mfccs[:,:max_pad_len])
    # print(mfcc_sh.shape)    
out[0]


# In[5]:


len(out)


# In[6]:


seesid_li = []
label_li = []
for path in path_li:
    seesid = (path.split('/')[-1].split('.')[0])
    seesid_li.append(seesid)
seesid_df = pd.DataFrame(seesid_li)
seesid_df.columns = ['Segment ID']
seesid_df


# In[7]:


label_path = 'E:/Competition/ETRI Competition//Analysis/감정 레이블/train 감정_보이스'     # 감정레이블 경로설정
label_filename = os.listdir(label_path)                                                  # 경로 설정한 폴더 열기
label_li = [label_file for label_file in label_filename if label_file.endswith('.csv')]  # .csv 파일만 블러오기


# In[8]:


# 감정라벨
total_label_df = pd.DataFrame()  
for filename in label_li:
    temp_label_df = pd.read_csv(label_path+'/'+filename)  # 감정레이블 파일 내용 읽어드리기
    total_label_df = pd.concat([total_label_df, temp_label_df], axis=0) 

total_label_df = total_label_df.reset_index(drop=True) 
print(len(total_label_df))

total_label_df[0:5]


# In[9]:


# wav 파일 기준으로 음성파일과 segment id가 동일하다면 Total Evaluation Emotion값을 가져옴
label_value_li = []
for i in range(len(seesid_df)):
   for k in range(len(total_label_df)):
           if seesid_df['Segment ID'][i] == total_label_df['Segment ID'][k]:
               label_value_li.append(total_label_df.loc[k, 'Total Evaluation Emotion'])


# In[30]:


train_label=pd.DataFrame(label_value_li)
train_label
train_label.to_csv("final_train_label0854.csv", index = False)


# In[4]:


train_label= pd.read_csv('E:/Competition/ETRI Competition/Analysis/final_train_label0854.csv')
train_label


# In[10]:


len(label_value_li)


# In[5]:


x_train = np.array(out)
x_train.shape


# In[6]:


y = np.array(train_label)
y.shape


# In[7]:


le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y))
y_train.shape


# ## CNN model 학습

# In[11]:


n_columns = 3300    
n_row = 128       
n_channels = 1
n_classes = 7

# input shape 조정
x_train_input = tf.reshape(x_train, [-1, n_row, n_columns, n_channels])


# In[12]:


model = keras.Sequential()

model.add(layers.Conv2D(input_shape=(n_row, n_columns, n_channels), filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(kernel_size=2, filters=128, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(kernel_size=2, filters=128, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(kernel_size=2, filters=256, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.1))

model.add(layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(units=7, activation='softmax'))

model.summary()


# In[14]:


es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)  # val_loss값이 10번정도 안떨어지면 학습 멈추게 하는 코드
mc = ModelCheckpoint('aaaaa.h5', monitor='loss', mode='min', save_best_only=True)


# In[15]:


training_epochs = 15
num_batch_size = 64

learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train_input, y_train, batch_size=num_batch_size, epochs=training_epochs, callbacks=[es, mc])


# In[38]:


# 모델을 저장한 파일 경로 설정
model_path = 'E:/Competition/ETRI Competition/Analysis/final_model.h5'


# In[41]:


# 모델 로드
audio_model = load_model(model_path)
audio_model


# ## test 예측

# In[16]:


path_t = 'E:/Competition/ETRI Competition/RAW DATA/KEMDy19/wav/test'    #데이터폴더 경로 지정해주기 
parent_folders_t = os.listdir(path_t)                     #폴더 불러오기
path_li_t = []
total_sum_t = 0
for parent_folder_t in parent_folders_t:
    child_folders_t = []
    child_folders_t.append((np.array(os.listdir(path_t+"/"+parent_folder_t)).flatten().tolist())) # wav에 있는 폴더 다 불러오기
    temp_folders_t = (np.array(child_folders_t)).flatten().tolist()
    for folder_t in temp_folders_t:
        files_t = os.listdir(path_t+"/"+parent_folder_t+"/"+folder_t) #sessio01폴더 불러오기
        for file_t in files_t:
            if (path_t+"/"+parent_folder_t+'/'+folder_t+'/'+file_t).endswith('.wav') == True: # session01_impro01안에있는 txt 파일만 순서대로 path_li로 보내기
                path_li_t.append(path_t+"/"+parent_folder_t+"/"+folder_t+"/"+file_t)

path_li_t[:3]


# In[17]:


len(path_li_t)


# ### 14-45초 음성 파일을 10초자리로 데이터 shape 맞춰주기

# In[61]:


# 14초이상 되는 음성 파일들을 가져옴
audio_files_t = []
for filename_t in path_li_t: # 음성 파일이 저장된 폴더 경로
    y_t, sr_t = librosa.load(filename_t, sr=16000)
    duration_t = librosa.get_duration(y=y_t, sr=sr_t)
    if duration_t < 14:
        continue
    audio_files_t.append(filename_t)


# In[62]:


len(audio_files_t)  #14초이상 124


# In[55]:


# 14초~45초 되는 음성 파일들을 저장할 폴더 경로
slid_test = 'E:/Competition/ETRI Competition/RAW DATA/KEMDy19/wav/test/slid_test'


# In[56]:


segment_duration = 10

for file_path_t in audio_files_t:
    filename_t = os.path.basename(file_path_t)
    y_t, sr_t = librosa.load(file_path_t, sr=16000, offset=0)
    num_segments = int(np.ceil(len(y) / (segment_duration * sr)))
    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        y_segment = y[int(start * sr):int(end * sr)]
        output_file_t = os.path.join(slid_test, f'{filename_t}.wav')
        sf.write(output_file_t, y_segment, sr_t, subtype='PCM_16')


# In[57]:


for file_delete_t in audio_files_t:  #14초이상 45초 미만 음성파일 삭제
    os.remove(file_delete_t)


# ## test 예측

# In[18]:


max_pad_len_t = 3300
hop_length_t = 160  # 전체 frame 수 (hop_length의 길이만큼 옆으로 가면서 데이터를 읽는 옵션, 10ms를 기본으로 하고 있어 16000Hz인 음성에서는 160에 해당)
n_fft_t = 400       # frame 하나당 sample 수(일반적으로 자연어 처리에서는 음성을 25m의 크기를 기본으로 하고 있으며 16000Hz인 음성에서는 400에 해당하는 값) 

out_t = []
for file_t in path_li_t:
    data_t, sr_t = librosa.load(file_t)  # librosa로 음성 데이터 불러오기
    mfccs_t = librosa.feature.mfcc(y=data_t, sr=sr_t, n_fft=n_fft_t, hop_length=hop_length_t, n_mfcc=128)  #mfcc로 음성을 array형태로 나타냄
    pad_width_t = max_pad_len_t - mfccs_t.shape[1]
    if pad_width_t < 0:
        mfcc_sh_t = mfccs_t[:, :max_pad_len_t]
    else:
        mfcc_sh_t = np.pad(mfccs_t, pad_width=((0, 0), (0, pad_width_t)), mode='constant')
    out_t.append(mfcc_sh_t)
out_t[0]


# In[19]:


len(out_t)


# In[20]:


seesid_li_t = []
label_li_t = []
for path_t in path_li_t:
    seesid_t = (path_t.split('/')[-1].split('.')[0])
    seesid_li_t.append(seesid_t)
seesid_df_t = pd.DataFrame(seesid_li_t)
seesid_df_t.columns = ['Segment ID']
seesid_df_t


# In[21]:


label_path_t = 'E:/Competition/ETRI Competition/Analysis/감정 레이블/test 감정_보이스'  # 감정레이블 경로설정
label_filename_t = os.listdir(label_path_t)                                  # 경로 설정한 폴더 열기
label_li_t = [label_file_t for label_file_t in label_filename_t if label_file_t.endswith('.csv')]  # .csv 파일만 블러오기


# In[22]:


# 감정라벨
total_label_df_t = pd.DataFrame()  
for filename_t in label_li_t:
    temp_label_df_t = pd.read_csv(label_path_t+'/'+filename_t)  # 감정레이블 파일 내용 읽어드리기
    total_label_df_t = pd.concat([total_label_df_t, temp_label_df_t], axis=0) 

total_label_df_t = total_label_df_t.reset_index(drop=True) 
print(len(total_label_df_t))

total_label_df_t[0:5]


# In[23]:


#wav와 label을 segment id기준으로 다른게 있는지 확인

label_value_li_t = []
non_li_t = []
non_seg_li_t = []
joungbok_li_t = []
for i_t in range(len(seesid_df_t)):
   check_t = 0
   for k_t in range(len(total_label_df_t)):
       if check_t == 0:
           if seesid_df_t['Segment ID'][i_t] == total_label_df_t['Segment ID'][k_t]:
               label_value_li_t.append(total_label_df_t['Total Evaluation Emotion'][k_t])
               check_t = 1
       elif check_t == 1:
           if seesid_df_t['Segment ID'][i_t] == total_label_df_t['Segment ID'][k_t]:
               joungbok_li_t.append(seesid_df_t['Segment ID'][i_t])
               check_t = 1
print(len(label_value_li_t))
print(len(joungbok_li_t))
print(joungbok_li_t)


# In[69]:


# 두 데이터프레임에서 공통으로 포함된 Segment ID를 찾습니다.
common_ids = seesid_df_t['Segment ID'].isin(total_label_df_t['Segment ID'])

# 두 데이터프레임에서 공통으로 포함되지 않은 Segment ID를 찾습니다.
different_ids = ~common_ids

# different_ids를 사용하여 seesid_df_t에서 공통되지 않은 Segment ID를 찾습니다.
unique_seesid = seesid_df_t.loc[different_ids, 'Segment ID']


# In[70]:


unique_seesid # 삭제


# In[24]:


x_test = np.array(out_t)
x_test.shape


# In[25]:


y_test1 = np.array(label_value_li_t)
y_test1.shape


# In[26]:


le1 = LabelEncoder()
arr_float = to_categorical(le1.fit_transform(y_test1))
y_test = arr_float.astype(np.int64)
print(y_test.dtype)
print(arr_float.shape)


# In[28]:


y_test


# ## 예측모델 평가

# In[42]:


predictions = audio_model.predict(x_test)


# In[51]:


real_label_li = []
pred_label_li = []
for i in range(len(y_test)):
    real_label_li.append(np.argmax(y_test[i]))
    pred_label_li.append(np.argmax(predictions[i]))


# In[49]:


# 예측값을 데이터프레임으로 변환
#df = pd.DataFrame({'Prediction': pred_label_li})
#df.to_csv('voice_predictions.csv', index=False)

#print(df[:5])


# In[52]:


counter = 0 
total = 0 
for k in range(len(real_label_li)):
    total += 1
    if real_label_li[k] == pred_label_li[k]:
        counter += 1
    else:
        pass

print("Accuracy: ", counter / total)


# In[33]:


import tensorflow_addons as tfa


# In[34]:


metric = tfa.metrics.F1Score(num_classes=7,average='macro', threshold=None, name='f1_score', dtype=None)


# In[35]:


metric.update_state(y_test, predictions)
result = metric.result()
#result.numpy()
print("f1_score: ", result.numpy())

