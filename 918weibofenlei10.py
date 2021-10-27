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
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#文件读取以及转换
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

texts = []
labelss=[]
with open("10w.csv", "r", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    mems = [mem for mem in reader]
    texts = [row["review"] for row in mems]
    labelss=[row["label"] for row in mems]
    labels=[int(x) for x in labelss]
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, shuffle=True)

# train_texts,val_texts,train_labels,val_labels=train_test_split(train_texts,train_labels,test_size=0.2)


tokenizer=BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')
train_encodings=tokenizer(train_texts,truncation=True,padding=True)
# val_encodings=tokenizer(val_texts,truncation=True,padding=True)
test_encodings=tokenizer(test_texts,truncation=True,padding=True)



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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext',num_labels=2)
print(model)
model=nn.DataParallel(model,device_ids=[i for i in range(4)])
model.cuda()
optim = AdamW(model.parameters(), lr=1e-5)
model.to(device)
model.train()
batch_size=64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size)
print(len(test_loader))



def test(class_correct,class_total,predictLabel,epoch):
    # model.eval()
    total=0
    correct=0
    k=0
    with torch.no_grad():
        ii=0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output=model(input_ids, attention_mask=attention_mask, labels=labels)
            _,predicted=torch.max(output.logits,1)
            for j in range(len(predicted)):
                predictLabel[k] = predicted[j]
                k += 1
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            # test_loss+=F.cross_entropy(output,labels.squeeze(),size_average=False).item()
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            print("In  this %d test batch: Accuracy:   %.3f %% ,  Loss:   %.6f"%(ii+1,100*(predicted==labels).sum().item()/batch_size,output[0].mean()))
            ii+=1
        accuracy=int(100*correct/total)
        print('correctly number ：   %d   ，Accuracy of the network :  %.6f%%' %(correct,100*correct/total))
        date=time.strftime("%Y%m%d%H%M", time.localtime())
        path = 'model/epoch' + str(epoch) + 'model'+str(date)+str(accuracy)+'.pth'
        torch.save(model, path)

for epoch in range(6):
    running_loss=0.0
    time_begin = time.time()
    for batch_idx,batch in enumerate(train_loader,0):

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0].mean()
        running_loss += loss.item()
        loss.backward()
        optim.step()
        if batch_idx%10==9:
            _, predict = torch.max(outputs.logits, 1)
            correct_num = (predict == labels).sum().item()
            accuracy = correct_num / len(predict)
            time_end=time.time()
            print("Epoch:{}      [{}/{}({:.0f}%)]\tTraining Loss: {:.6f}  tp:{:.2f}%  consuming time：{:.3f}".format(
                epoch,batch_idx+1,len(train_loader.dataset)/batch_size,100.*batch_idx/len(train_loader),
                running_loss/10,100*accuracy,time_end-time_begin
            ))
            running_loss=0.0
            time_begin = time.time()

    # path='model/'+str(epoch)+'my_model.pth'
    # torch.save(model.state_dict(),path)
    # path = 'model/' + str(epoch) + 'model9171357.pth'
    # torch.save(model, path)
    class_correct = np.zeros(4)
    class_total = np.zeros(4)

    predictLabel = np.zeros(len(test_dataset))
    #只是保存参数，不保存模型本身。而且不能赋值，不能model=model.load_state_dict(torch.load('model/0my_model.pth'))，这样也会报错，这是一种方式、
    # #另外一种是在保存模型时直接保存模型本身，model.save(model,path)
    # model.load_state_dict(torch.load('model/0my_model.pth'))
    model.eval()
    test( class_correct,class_total,predictLabel,epoch)





