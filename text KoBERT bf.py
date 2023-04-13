#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install mxnet')
get_ipython().system('pip install gluonnlp pandas tqdm')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install transformers==3.0.2')
get_ipython().system('pip install torcheval')
get_ipython().system('pip install git+https://git@github.com/SKTBrain/KoBERT.git@master')
get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import glob
import itertools
import os

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


#GPU 사용
device = torch.device("cuda:0")


# In[ ]:


path = '/content/drive/MyDrive/공모전_ETRI/wav/train'    #데이터폴더 경로 지정해주기 
parent_folders = os.listdir(path)                     #폴더 불러오기
path_li = []
for parent_folder in parent_folders:
    child_folders = []
    child_folders.append((np.array(os.listdir(path+"/"+parent_folder)).flatten().tolist())) # wav에 있는 폴더 다 불러오기
    temp_folders = (np.array(child_folders)).flatten().tolist()
    for folder in temp_folders:
        files = os.listdir(path+"/"+parent_folder+"/"+folder) #sessio01폴더 불러오기
        for file in files:
            if (path+"/"+parent_folder+folder+file).endswith('.txt') == True: # session01_impro01안에있는 txt 파일만 순서대로 path_li로 보내기
                path_li.append(path+"/"+parent_folder+"/"+folder+"/"+file)
path_li[:3]


# In[ ]:


print(len(path_li))


# In[ ]:


df= []
for i in path_li:
        with open(i, 'r') as f:    # 텍스트 파일 읽어드리기
            data= f.read()#.to(device)
            df.append(data)
df = pd.DataFrame(df)

temp_path_li = []
for path in path_li:
    temp_path_li.append(path.split('.')[0].split('/')[-1])  # 파일이름을 segment id 이름 행 값으로 표현하도록 전처리 작업
file_list_df = pd.DataFrame(temp_path_li)
converting_df = pd.concat([file_list_df, df], axis=1) # 하나씩 불러온 text와 segment id를 결합
print(len(converting_df))

converting_df.columns=['Segment ID', 'Text'] # 변수이름 지정
print(converting_df.duplicated(['Segment ID']).value_counts())
converting_df
# 8187


# In[ ]:


label_path = '/content/drive/MyDrive/공모전_ETRI/train_before'  # 감정레이블 경로설정
label_filename = os.listdir(label_path)                                  # 경로 설정한 폴더 열기
label_li = [label_file for label_file in label_filename if label_file.endswith('.csv')]  # .csv 파일만 블러오기


# In[ ]:


total_label_df = pd.DataFrame()  
for filename in label_li:
    temp_label_df = pd.read_csv(label_path+'/'+filename)  # 감정레이블 파일 내용 읽어드리기
    total_label_df = pd.concat([total_label_df, temp_label_df], axis=0) 
total_label_df = total_label_df.reset_index(drop=True) 
print(len(total_label_df))
total_label_df.head(3)
#8725 중복 포함


# In[ ]:


len(total_label_df)


# In[ ]:


total_label_df['Total Evaluation Emotion'].value_counts()


# In[ ]:


common_segments = set(total_label_df['Segment ID']).intersection(set(converting_df['Segment ID'])) #total_label_df와 converting_df의 Segment ID 열에 존재하는 공통된 값만 추출

for segment_id in common_segments:
    text = converting_df.loc[converting_df['Segment ID'] == segment_id, 'Text'].values[0]
    total_label_df.loc[total_label_df['Segment ID'] == segment_id, 'Text'] = text


# In[ ]:


print(total_label_df['Text'].isna().sum())


# In[ ]:


nan_rows = total_label_df[total_label_df['Text'].isna()]
nan_rows1 = pd.DataFrame(nan_rows)
nan_rows1 # 감정 라벨은 있지만 text가 없는 segment id


# In[ ]:


merge_df=total_label_df.dropna(subset = ['Text'])
merge_df


# In[ ]:


print(merge_df['Text'].isna().sum())


# In[ ]:


# # angry = 0
# # disgust = 1
# # fear = 2
# # happy = 3
# # neutral = 4
# # sad = 5
# # surprise = 6

merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'angry'),'Total Evaluation Emotion'] = '0'
merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'disgust'),'Total Evaluation Emotion'] = '1'
merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'fear'),'Total Evaluation Emotion'] = '2'
merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'happy'),'Total Evaluation Emotion'] = '3'
merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'neutral'), 'Total Evaluation Emotion'] = '4'
merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'sad'),'Total Evaluation Emotion'] = '5'
merge_df.loc[(merge_df['Total Evaluation Emotion'] == 'surprise'),'Total Evaluation Emotion'] = '6'

merge_df.loc[(merge_df['Before Emotion'] == 'angry'),'Before Emotion'] = '분노'
merge_df.loc[(merge_df['Before Emotion'] == 'disgust'),'Before Emotion'] = '혐오'
merge_df.loc[(merge_df['Before Emotion'] == 'fear'),'Before Emotion'] = '두려움'
merge_df.loc[(merge_df['Before Emotion'] == 'happy'),'Before Emotion'] = '행복한'
merge_df.loc[(merge_df['Before Emotion'] == 'neutral'), 'Before Emotion'] = '중립'
merge_df.loc[(merge_df['Before Emotion'] == 'sad'),'Before Emotion'] = '슬픔'
merge_df.loc[(merge_df['Before Emotion'] == 'surprise'),'Before Emotion'] = '놀람'


# In[ ]:


merge_df[:5]


# In[ ]:


merge_df['before_text']= merge_df['Text'] + ' ' + merge_df['Before Emotion']
before_text = merge_df


# In[ ]:


#before_text.to_csv('before_emotion.csv', index= False)


# In[ ]:


merge_df=pd.read_csv('/content/drive/MyDrive/공모전_ETRI/new_model_directory/model22.pt',encoding='utf-8',sep=',' )
merge_df


# In[ ]:


data_li = []
for q, label in zip(merge_df['Text'], merge_df['Total Evaluation Emotion']):
    data = []
    data.append(q)
    data.append(str(label))
    
    data_li.append(data)


# In[ ]:


data_li[:5]


# In[ ]:


cd /content/drive/MyDrive/ris 공모전/KoBERT


# In[ ]:


#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
#kobert
from kobert.utils.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from tqdm import tqdm, tqdm_notebook


# In[ ]:


#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()


# In[ ]:


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                pad, pair, mode = "train"):
        self.mode = mode
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length = max_len, pad = pad, pair = pair)
        if self.mode == "train":
            self.sentences = [transform([i[sent_idx]]) for i in dataset]
            self.labels = [np.int32(i[label_idx]) for i in dataset]
            
        else:
            self.sentences = [transform(i) for i in dataset]
        
    def __getitem__(self, i):
        if self.mode == 'train':
            return (self.sentences[i] + (self.labels[i], ))
        else:
            return self.sentences[i]
    def __len__(self):
        return (len(self.sentences))


# In[ ]:


max_len = 0 
for i in range(len(data_li)):
    if len(data_li[i][0]) > max_len:
        max_len = len(data_li[i][0])

max_len


# In[ ]:


# Setting parameters
max_len = 318
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


# In[ ]:


#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data_train = BERTDataset(data_li, 0, 1, tok, max_len, True, False, mode="train")
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)


# In[ ]:


data_train


# In[ ]:


train_dataloader


# In[ ]:


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768, #768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# In[ ]:


#BERT 모델 불러오기
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
 
#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


# In[ ]:


#epoch 10
for e in range(num_epochs):
    train_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids).to(device)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))


# In[ ]:


# 새로운 디렉토리 생성
os.makedirs('/content/drive/MyDrive/공모전_ETRI/new_model_directory', exist_ok=True)

# 모델 저장
model_path = '/content/drive/MyDrive/공모전_ETRI/new_model_directory/before_model.pt'
torch.save(model, model_path)


# In[ ]:


model_path = '/content/drive/MyDrive/공모전_ETRI/new_model_directory/before_model.pt'
model = torch.load(model_path)  #BERTClassifie 함수를 실행해야 모델load가 가능함.
model


# ## TEST data set 만들기

# In[ ]:


path_t = '/content/drive/MyDrive/공모전_ETRI/wav/test'    #데이터폴더 경로 지정해주기 
parent_folders_t = os.listdir(path_t)                     #폴더 불러오기
path_li_t = []
for parent_folder_t in parent_folders_t:
    child_folders_t = []
    child_folders_t.append((np.array(os.listdir(path_t+"/"+parent_folder_t)).flatten().tolist())) # wav에 있는 폴더 다 불러오기
    temp_folders_t = (np.array(child_folders_t)).flatten().tolist()
    for folde_tr_t in temp_folders_t:
        files_t = os.listdir(path_t+"/"+parent_folder_t+"/"+folde_tr_t) #sessio01폴더 불러오기
        for file_t in files_t:
            if (path_t+"/"+parent_folder_t+folde_tr_t+file_t).endswith('.txt') == True: # session01_impro01안에있는 txt 파일만 순서대로 path_li로 보내기
                path_li_t.append(path_t+"/"+parent_folder_t+"/"+folde_tr_t+"/"+file_t)
path_li_t[:3]


# In[ ]:


print(len(path_li_t))


# In[ ]:


df_t= []
for i in path_li_t:
        with open(i, 'r') as f:    # 텍스트 파일 읽어드리기
            data_t= f.read()
            df_t.append(data_t)
df_t = pd.DataFrame(df_t)

temp_path_li_t = []
for path_t in path_li_t:
    temp_path_li_t.append(path_t.split('.')[0].split('/')[-1])  # 파일이름을 segment id 이름 행 값으로 표현하도록 전처리 작업
file_list_df_t = pd.DataFrame(temp_path_li_t)
converting_df_t = pd.concat([file_list_df_t, df_t], axis=1) # 하나씩 불러온 text와 segment id를 결합
print(len(converting_df_t))

converting_df_t.columns=['Segment ID', 'Text'] # 변수이름 지정
print(converting_df_t.duplicated(['Segment ID']).value_counts())
converting_df_t


# In[ ]:


label_path_t = '/content/drive/MyDrive/공모전_ETRI/test_before'  # 감정레이블 경로설정
label_filename_t = os.listdir(label_path_t)                                  # 경로 설정한 폴더 열기
label_li_t = [label_file_t for label_file_t in label_filename_t if label_file_t.endswith('.csv')]  # .csv 파일만 블러오기


# In[ ]:


total_label_df_t = pd.DataFrame()  
for filename_t in label_li_t:
    temp_label_df_t = pd.read_csv(label_path_t+'/'+filename_t)  # 감정레이블 파일 내용 읽어드리기
    total_label_df_t = pd.concat([total_label_df_t, temp_label_df_t], axis=0) 
total_label_df_t = total_label_df_t.reset_index(drop=True) 
print(len(total_label_df_t))
total_label_df_t.head(3)


# In[ ]:


common_segments_t = set(total_label_df_t['Segment ID']).intersection(set(converting_df_t['Segment ID'])) #total_label_df와 converting_df의 Segment ID 열에 존재하는 공통된 값만 추출

for segment_id_t in common_segments_t:
    text_t = converting_df_t.loc[converting_df_t['Segment ID'] == segment_id_t, 'Text'].values[0]
    total_label_df_t.loc[total_label_df_t['Segment ID'] == segment_id_t, 'Text'] = text_t


# In[ ]:


total_label_df_t


# In[ ]:


print(total_label_df_t['Text'].isna().sum())


# In[ ]:


nan_rows_t = total_label_df_t[total_label_df_t['Text'].isna()]
nan_rows1_t = pd.DataFrame(nan_rows_t)
nan_rows1_t # 감정 라벨은 있지만 text가 없는 segment id


# In[ ]:


merge_test=total_label_df_t.dropna(subset = ['Text'])
merge_test


# In[ ]:


merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'angry'),'Total Evaluation Emotion'] = '0'
merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'disgust'),'Total Evaluation Emotion'] = '1'
merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'fear'),'Total Evaluation Emotion'] = '2'
merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'happy'),'Total Evaluation Emotion'] = '3'
merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'neutral'), 'Total Evaluation Emotion'] = '4'
merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'sad'),'Total Evaluation Emotion'] = '5'
merge_test.loc[(merge_test['Total Evaluation Emotion'] == 'surprise'),'Total Evaluation Emotion'] = '6'

merge_test.loc[(merge_test['Before Emotion'] == 'angry'),'Before Emotion'] = '분노'
merge_test.loc[(merge_test['Before Emotion'] == 'disgust'),'Before Emotion'] = '혐오'
merge_test.loc[(merge_test['Before Emotion'] == 'fear'),'Before Emotion'] = '두려움'
merge_test.loc[(merge_test['Before Emotion'] == 'happy'),'Before Emotion'] = '행복한'
merge_test.loc[(merge_test['Before Emotion'] == 'neutral'), 'Before Emotion'] = '중립'
merge_test.loc[(merge_test['Before Emotion'] == 'sad'),'Before Emotion'] = '슬픔'
merge_test.loc[(merge_test['Before Emotion'] == 'surprise'),'Before Emotion'] = '놀람'


# In[ ]:


merge_test['before_text']= merge_test['Text'] + ' ' + merge_test['Before Emotion']
before_test = merge_test


# In[ ]:


before_test[:5]


# In[ ]:


before_test.to_csv('before_test.csv', index= False)


# In[ ]:


before_test=pd.read_csv('/content/drive/MyDrive/공모전_ETRI/before_test.csv',encoding='utf-8',sep=',' ) #merge_test


# ##KoBERT test Input dataset

# In[ ]:


test_data=[]
for q, label in zip(before_test['before_text'],before_test['Total Evaluation Emotion']):
    data_1 =[]
    data_1.append(q)
    data_1.append(str(label))

    test_data.append(data_1)


# In[ ]:


test_data[:5]


# In[ ]:


another_test = BERTDataset(test_data, 0, 1, tok, max_len, True, False)
test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)


# In[ ]:


#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc, max_indices


# ##test predict

# In[ ]:


from torcheval.metrics.functional import multiclass_f1_score


# In[ ]:


# for e in range(num_epochs):
test_acc = 0.0 
f1_score = 0.0
prediction = []
ori_label = []

model.eval()
for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)

    #calc_accuracy
    max_vals, max_indices = torch.max(out, 1)
    test_acc += (max_indices == label).sum().data.cpu().numpy()/max_indices.size()[0]

    f1_score += multiclass_f1_score(out, label, num_classes=7, average="macro")

    prediction.append(max_indices.data.cpu().numpy())
    ori_label.append(label.data.cpu().numpy())

print("epoch {} f1-score {}".format(1, f1_score / (batch_id+1)))
print("epoch {} test acc {}".format(1, test_acc / (batch_id+1)))


# In[ ]:


print(prediction)
print(ori_label)


# In[ ]:


pred_0 = 0
pred_fault_0 = 0
pred_1 = 0
pred_fault_1 = 0
pred_2 = 0
pred_fault_2 = 0
pred_3 = 0
pred_fault_3 = 0
pred_4 = 0
pred_fault_4 = 0
pred_5 = 0
pred_fault_5 = 0
pred_6 = 0
pred_fault_6 = 0

for i in range(len(prediction)):
  for k in range(len(prediction[i])):
      if prediction[i][k] == 0:
        if prediction[i][k] == ori_label[i][k]:
          pred_0 += 1
        else:
          pred_fault_0 += 1

      elif prediction[i][k] == 1:
        if prediction[i][k] == ori_label[i][k]:
          pred_1 += 1
        else:
          pred_fault_1 += 1

      elif prediction[i][k] == 2:
        if prediction[i][k] == ori_label[i][k]:
          pred_2 += 1
        else:
          pred_fault_2 += 1

      elif prediction[i][k] == 3:
        if prediction[i][k] == ori_label[i][k]:
          pred_3 += 1
        else:
          pred_fault_3 += 1

      elif prediction[i][k] == 4:
        if prediction[i][k] == ori_label[i][k]:
          pred_4 += 1
        else:
          pred_fault_4 += 1

      elif prediction[i][k] == 5:
        if prediction[i][k] == ori_label[i][k]:
          pred_5 += 1
        else:
          pred_fault_5 += 1

      elif prediction[i][k] == 6:
        if prediction[i][k] == ori_label[i][k]:
          pred_6 += 1
        else:
          pred_fault_6 += 1


# In[ ]:


print("Label 0 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_0/(pred_0 + pred_fault_0), (pred_0 + pred_fault_0), pred_fault_0))
print("Label 1 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_1/(pred_1 + pred_fault_1), (pred_1 + pred_fault_1), pred_fault_1))
print("Label 2 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_2/(pred_2 + pred_fault_2), (pred_2 + pred_fault_2), pred_fault_2))
print("Label 3 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_3/(pred_3 + pred_fault_3), (pred_3 + pred_fault_3), pred_fault_3))
print("Label 4 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_4/(pred_4 + pred_fault_4), (pred_4 + pred_fault_4), pred_fault_4))
print("Label 5 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_5/(pred_5 + pred_fault_5), (pred_5 + pred_fault_5), pred_fault_5))
print("Label 6 accuracy: {} // {} 중 오탐 갯수 {}".format(pred_6/(pred_6 + pred_fault_6), (pred_6 + pred_fault_6), pred_fault_6))


# In[ ]:




