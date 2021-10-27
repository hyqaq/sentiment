import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel, BertTokenizerFast
import json
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification,Trainer,TrainingArguments
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from torch.optim import SGD
import numpy as np
import os
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
# class BERT_LSTM(nn.Module):
#     def __init__(self,class_num,dim,bert_dir='bert-base-uncased',num_layer=1):
#         super(BERT_LSTM,self).__init__()
#         self.class_num=class_num
#         self.bert_encoder=BertModel.from_pretrained(bert_dir)
#         my_config=BertConfig(**config)
###############模型
class bert_lstm(nn.Module):
    def __init__(self,hidden_dim,output_size,n_layers,bidirection=True,drop=0.5):
        super(bert_lstm,self).__init__()
        self.hidden_dim=hidden_dim
        self.output_size=output_size
        self.n_layers=n_layers
        self.bidirection=bidirection
        self.bert=BertModel.from_pretrained('bert-base-cased')
        self.lstm=nn.LSTM(768,hidden_dim,n_layers,batch_first=True,bidirectional=bidirection)
        self.dropout=nn.Dropout(drop)
        if bidirection==True:
            self.fc=nn.Linear(hidden_dim*2,output_size)
        else:
            self.fc=nn.Linear(hidden_dim,output_size)
    def forward(self,x,hidden):
        batch_size=x.size(0)
        x=self.bert(x)[0]
        lstm_out,(hidden_last,cn_last)=self.lstm(x,hidden)
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        out = self.fc(out)
        return out

class bertModel(nn.Module):
    def __init__(self):
        super(bertModel,self).__init__()
        self.bert=BertModel.from_pretrained('bert-base-cased')
        # self.l1=nn.Linear(768,256)
        # self.l2=nn.Linear(256,128)
        # self.l3=nn.Linear(128,2)
        # self.lstm = nn.LSTM()
        self.fc=nn.Linear(768,2)

    def forward(self,ids,mask):
        result=self.bert(ids, mask)
        # sequence_output,pooled_output = result[0],result[1]
        sequence_output = result[0]
        # print(sequence_output.size())
        # print(self.bert(ids,mask))
        # l1_output=self.l1(sequence_output[:,0,:].view(-1,768))
        # # l1_output=F.relu(l1_output)
        # l2_output=self.l2(l1_output)
        # l2_output=F.relu(l2_output)
        # l3_output=self.l3(l2_output)
        # l3_output=F.relu(l3_output)
        l1_output=self.fc(F.relu(sequence_output[:,0,:].view(-1,768)))
        return l1_output


###################文件读取以及转换
def read_imdb(split_dir):
    split_dir=Path(split_dir)
    texts=[]
    labels=[]
    # print(split_dir.is_dir())
    for label_dir in["pos","neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            # text_file.open(mode='r', buffering = -1, encoding = 'gbk', errors = None, newline = None)
            #这里制定编码方式，否则报错UnicodeDecodeError: 'gbk' codec can't decode byte 0x93 in position 596: illegal multibyte sequence
            texts.append(text_file.read_text(encoding="utf8"))
            labels.append(0 if label_dir == "neg" else 1)

    return texts,labels


train_texts,train_labels=read_imdb('/home/lh1926/aclImdb/train')
test_texts,test_labels=read_imdb('/home/lh1926/aclImdb/test')


train_texts,val_texts,train_labels,val_labels=train_test_split(train_texts,train_labels,test_size=0.2)


tokenizer=BertTokenizerFast.from_pretrained('bert-base-cased')
train_encodings=tokenizer(train_texts,truncation=True,padding=True)
val_encodings=tokenizer(val_texts,truncation=True,padding=True)
test_encodings=tokenizer(test_texts,truncation=True,padding=True)

# print(test_encodings)

#将数据集转为pytorch的数据集
class IMDBdata(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels

    def __getitem__(self, idx):
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return  len(self.labels)

train_dataset=IMDBdata(train_encodings,train_labels)
val_dataset=IMDBdata(val_encodings,val_labels)
test_dataset=IMDBdata(test_encodings,test_labels)

print(len(test_dataset))
# #训练参数设定，使用Trainer的方法，
# train_args=TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10
# )
# #model 使用Trainer的方法，
# model=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
# trainer=Trainer(
#     model=model,
#     args=training_args,  # training arguments, defined above
#     train_dataset=train_dataset,  # training dataset
#     eval_dataset=val_dataset  # evaluation dataset
# )
# trainer.train()
#
#

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
loss_fn=nn.CrossEntropyLoss()

#####指定多路gpu训练
model=bertModel()
model = nn.DataParallel(model)
model = model.cuda()
print(model)
optim = AdamW(model.parameters(), lr=5e-5)
model.to(device)
model.train()
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size)
print(len(test_loader))

# optim = SGD(model.parameters(), lr=5e-3)

def test(class_correct,class_total,predictLabel):
    # model.eval()
    total=0
    correct=0
    k=0
    with torch.no_grad():
        for batch in test_loader:
            # data,target=data.cuda(),target.cuda()
            # inputs,labels=data
            # data,target=Variable(data,volatile=True),Variable(target)
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output=model(input_ids, mask=attention_mask)
            # print(output)
            _,predicted=torch.max(output.data,1)
            # print(predicted)
           # print(predicted,labels)
            for j in range(len(predicted)):
                predictLabel[k] = predicted[j]
                k += 1
            c = (predicted == labels).squeeze()
            # print(labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            # test_loss+=F.cross_entropy(output,labels.squeeze(),size_average=False).item()
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            # if (pred==target):
            #     print(pred)
        print('correctly number ：   %d   ，Accuracy of the network :  %.6f%%' %(correct,100*correct/total))


for epoch in range(3):
    running_loss=0.0
    for batch_idx,batch in enumerate(train_loader,0):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, mask=attention_mask)
        label = labels.view(1, -1)
        loss = loss_fn(outputs, label.squeeze())
        # loss = outputs[0]
        running_loss += loss.item()
        loss.backward()
        optim.step()
        if batch_idx%10==9:
            print("Epoch:{}      [{}/{}({:.2f}%)]\tTraining Loss: {:.6f},  train accuracy:{:.6f}".format(
                epoch+1,batch_idx+1,len(train_loader.dataset)/batch_size,100.*batch_idx/len(train_loader),
                running_loss/10,
                (outputs.argmax(dim=1)==label.squeeze()).sum(dim=0)/batch_size
            ))
            running_loss=0.0

    ###测试
    class_correct = np.zeros(2)
    class_total = np.zeros(2)
    predictLabel = np.zeros(len(test_dataset))
    #不太懂eval的作用
    model.eval()
    test( class_correct,class_total,predictLabel)