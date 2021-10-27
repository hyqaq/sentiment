# import torch
# import torch.nn as nn
# from transformers import BertModel,BertConfig
# import json
# import torch.nn.functional as F
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from transformers import DistilBertTokenizerFast
# import torch
# from transformers import DistilBertForSequenceClassification,Trainer,TrainingArguments
# from torch.utils.data import DataLoader
# from transformers import DistilBertForSequenceClassification, AdamW
# import numpy as np
# import os
# class NeuralNet(nn.Module):
#     def __init__(self, model_name_or_path, hidden_size=768, num_class=2):
#         super(NeuralNet, self).__init__()
#
#         self.config = BertConfig.from_pretrained(model_name_or_path, num_labels=2)
#         self.config.output_hidden_states = True#需要设置为true才输出
#         self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.weights = nn.Parameter(torch.rand(13, 1))
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(0.5) for _ in range(5)
#         ])
#         self.fc = nn.Linear(hidden_size, num_class)
#
#     def forward(self, input_ids, input_mask, segment_ids):
#         last_hidden_states, pool, all_hidden_states = self.bert(input_ids, token_type_ids=segment_ids,
#                                                                 attention_mask=input_mask)
#         batch_size = input_ids.shape[0]
#         ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
#             13, batch_size, 1, 768)
#         atten = torch.sum(ht_cls * self.weights.view(
#             13, 1, 1, 1), dim=[1, 3])
#         atten = F.softmax(atten.view(-1), dim=0)
#         feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
#         for i, dropout in enumerate(self.dropouts):
#             if i == 0:
#                 h = self.fc(dropout(feature))
#             else:
#                 h += self.fc(dropout(feature))
#         h = h / len(self.dropouts)
#         return h
#
#     l3_output.size()
#     torch.Size([8, 2])
#     l1_output.size()
#     torch.Size([8, 256])
#     l2_output.size()
#     torch.Size([8, 128])
#     sequence_output[:, 0, :].view(-1, 768).size()
#     torch.Size([8, 768])


import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel
import json
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast,BertTokenizerFast,BertTokenizer
import torch
from transformers import DistilBertForSequenceClassification,Trainer,TrainingArguments
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from torch.optim import SGD
import numpy as np
import os
# class BERT_LSTM(nn.Module):
#     def __init__(self,class_num,dim,bert_dir='bert-base-uncased',num_layer=1):
#         super(BERT_LSTM,self).__init__()
#         self.class_num=class_num
#         self.bert_encoder=BertModel.from_pretrained(bert_dir)
#         my_config=BertConfig(**config)

class bertModel(nn.Module):
    def __init__(self):
        super(bertModel,self).__init__()
        self.bert=BertModel.from_pretrained('bert-base-uncased')
        # self.l1=nn.Linear(768,256)
        # self.l2=nn.Linear(256,128)
        # self.l3=nn.Linear(128,2)
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
        l1_output=F.relu(self.fc(sequence_output[:,0,:].view(-1,768)))
        return l1_output

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


train_texts,train_labels=read_imdb('/home/lh1926/aclImdb/train')
test_texts,test_labels=read_imdb('/home/lh1926/aclImdb/test')


train_texts,val_texts,train_labels,val_labels=train_test_split(train_texts,train_labels,test_size=0.2)


tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased')
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
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
loss_fn=nn.CrossEntropyLoss()
#指定多路gpu训练
model=bertModel()
# model=DistilBertForSequenceClassification()
# model.classifier = nn.Linear(model.config.dim, 2)
model = nn.DataParallel(model)
model = model.cuda()

print(model)
model.to(device)
# model=nn.DataParallel(model,device_ids=[0,1,2,3])
model.train()
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size)
print(len(test_loader))
optim = AdamW(model.parameters(), lr=5e-5)
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
        print('correctly number ：   %d   ，Accuracy of the network :  %d%%' %(correct,100*correct/total))


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
            print("Epoch:{}      [{}/{}({:.0f}%)]\tTraining Loss: {:.6f}, accuracy:{:.6f}".format(
                epoch,batch_idx+1,len(train_loader.dataset)/batch_size,100.*batch_idx/len(train_loader),
                running_loss/10,
                (outputs.argmax(dim=1)==label.squeeze()).sum(dim=0)/batch_size
            ))
            running_loss=0.0
    class_correct = np.zeros(2)
    class_total = np.zeros(2)

    predictLabel = np.zeros(len(test_dataset))
    #不太懂eval的作用
    model.eval()
    test( class_correct,class_total,predictLabel)
