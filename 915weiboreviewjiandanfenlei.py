import csv
import pandas as pd
from transformers import BertTokenizerFast,BertForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import  DataLoader
from  sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
# with open("36ww.csv", "r", encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     mems = [mem for mem in reader]
#     texts = [row["review"] for row in mems]
#     labelss=[row["label"] for row in mems]
#     labels=[int(x) for x in labelss]
#     train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, shuffle=True)
#     test_data = pd.DataFrame(test_texts,columns=["review"])
#     test_data.to_csv("36test_data.csv",index=False)
#     train_data = pd.DataFrame(train_texts, columns=["review"])
#     train_data.to_csv("36train_data.csv", index=False)
#     test_label = pd.DataFrame(test_labels, columns=["label"])
#     test_label.to_csv("36test_label.csv", index=False)
#     train_label = pd.DataFrame(test_labels, columns=["label"])
#     train_label.to_csv("36train_label.csv", index=False)



# print('25456745')

# with open("weibo_review2.csv","r",encoding="utf-8") as f:
#     reader=csv.DictReader(f)
#     test_text=[row["评论内容"] for row in reader]
# data=pandas.DataFrame(test_text)
# new_data=data.drop_duplicates(keep='first')
# tokenizer=BertTokenizerFast.from_pretrained("bert-base-chinese")
# test_encoding=tokenizer(test_text,padding=True,truncation=True)
# # print(test_text)
# test_label=np.ones(3361)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model=BertForSequenceClassification.from_pretrained("bert-base-chinese")
# model=nn.DataParallel(model)
# model.cuda()
# model.to(device)
# model.load_state_dict(torch.load('model/0my_model.pth'))
# model.eval()
# # for i in range(3361):
# #     outputs=model(test_encoding[i])
# #     print(outputs)
# class reviewdata(torch.utils.data.Dataset):
#     def __init__(self,encodings,labels):
#         self.encodings=encodings
#         self.labels=labels
#
#     def __getitem__(self, idx):
#         item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
#         item['labels']=torch.tensor(self.labels[idx])
#         return item
#     def __len__(self):
#         return  len(self.labels)
# test_dataset=reviewdata(test_encoding,test_label)
# positive=0
# negative=0
# # for batch in enumerate(:
# #     input_ids=batch['input_ids'].to(device)
# #     attention_mask = batch['attention_mask'].to(device)
# test_loader=DataLoader(test_dataset, batch_size=16)
# with torch.no_grad():
#     for batch in test_loader:
#
#         # labels = batch['labels'].to(device)
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         # labels = batch['labels'].to(device)
#         # model(input_ids, attention_mask=attention_mask)
#         output = model(input_ids, attention_mask=attention_mask)
#         # print(output)
#         _, predicted = torch.max(output.logits, 1)
#         print(predicted,output.logits)
#         positive+=predicted.sum()
#
# print(positive,3361-positive)

import csv
from datetime import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification,Trainer,TrainingArguments
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import numpy as np
import os
import torch.nn as nn
import sklearn
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#文件读取以及转换
texts = []
labelss=[]
with open("36train_data.csv", "r", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    train_texts=[row["review"] for row in reader]
with open("36train_label.csv", "r", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    train_labelss = [row["label"] for row in reader]
    train_labels=[int(x) for x in train_labelss]
with open("36test_data.csv", "r", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    test_texts = [row["review"] for row in reader]
with open("36test_label.csv", "r", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    test_labelss = [row["label"] for row in reader]
    test_labels=[int(x) for x in test_labelss]

# train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, shuffle=True)
# pd.save_csv(pd.from_numpy(train_texts))

# train_texts,train_labels=read_imdb('/home/lh1926/aclImdb/train')
# test_texts,test_labels=read_imdb('/home/lh1926/aclImdb/test')

# train_texts,val_texts,train_labels,val_labels=train_test_split(train_texts,train_labels,test_size=0.2)

#token化
tokenizer=BertTokenizerFast.from_pretrained('bert-base-chinese')
train_encodings=tokenizer(train_texts,truncation=True,padding=True)
# val_encodings=tokenizer(val_texts,truncation=True,padding=True)
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
# val_dataset=IMDBdata(val_encodings,val_labels)
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=2)
print(model)
model=nn.DataParallel(model)
model.cuda()
optim = AdamW(model.parameters(), lr=1e-5)
model.to(device)
model.train()
# path = 'model/' + str(10) + 'my_model.pth'
# torch.save(model.state_dict(), path)
batch_size=8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size)
print(len(test_loader))



def test(class_correct,class_total,predictLabel):
    # model.eval()
    total=0
    correct=0
    k=0
    with torch.no_grad():
        running_loss=0
        for batch_idx,batch in enumerate(test_loader,0):
            # data,target=data.cuda(),target.cuda()
            # inputs,labels=data
            # data,target=Variable(data,volatile=True),Variable(target)
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels = batch['labels'].to(device)
            output=model(input_ids, attention_mask=attention_mask, labels=labels)
            # print(output)
            _,predicted=torch.max(output.logits,1)
            loss = outputs[0].mean()
            running_loss += loss.item()
            # print(predicted)
           # print(predicted,labels)
            if batch_idx % 10 == 9:
                # time_end = time.time()
                print("[{}/{}({:.0f}%)]\tTraining Loss: {:.6f}  Acuuracy:{:.2f}".format(
                     batch_idx + 1, len(test_loader.dataset) / batch_size, 100. * batch_idx / len(test_loader),
                           running_loss / 10,(predicted==labels).sum().item()/batch_size
                ))
                running_loss = 0.0
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
    time_begin = time.time()
    for batch_idx,batch in enumerate(train_loader,0):

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs.logits, 1)
        loss = outputs[0].mean()
        running_loss += loss.item()
        loss.backward()
        optim.step()

        if batch_idx%10==9:
            time_end=time.time()
            print("Epoch:{}      [{}/{}({:.0f}%)]\tTraining Loss: {:.6f} Accuracy:{:.2f} consuming time：{:.3f}".format(
                epoch,batch_idx+1,len(train_loader.dataset)/batch_size,100.*batch_idx/len(train_loader),
                running_loss/10,(predicted==labels).sum().item()/batch_size,time_end-time_begin
            ))
            running_loss=0.0
            time_begin = time.time()
    path = 'model/' + str(epoch) + 'my_model9152259.pth'
    torch.save(model,path)
    class_correct = np.zeros(2)
    class_total = np.zeros(2)
    predictLabel = np.zeros(len(test_dataset))
    # #只是保存参数，不保存模型本身。而且不能赋值，不能model=model.load_state_dict(torch.load('model/0my_model.pth'))，这样也会报错，这是一种方式、
    # #另外一种是在保存模型时直接保存模型本身，model.save(model,path)
    # model.load_state_dict(torch.load('model/0my_model.pth'))
    # model.eval()
    test( class_correct,class_total,predictLabel)

# model.eval()



#模型评估


# print(test_dataset[0])

# print(train_texts)
# print(train_labels)
