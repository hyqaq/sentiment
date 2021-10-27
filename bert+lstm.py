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

USE_CUDA = torch.cuda.is_available()

class bert_lstm(nn.Module):
    def __init__(self, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert = BertModel.from_pretrained("bert-base-cased")
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden0,hidden1):
        batch_size = x.size(0)
        # 生成bert字向量
        x = self.bert(x)[0]  # bert 字向量

        # lstm_out
        # x = x.float()
        # hidden.view(4,8,-1)
        # print(hidden[0].shape,hidden[1].shape)

        lstm_out, (hidden_last, cn_last) = self.lstm(x, (hidden0.view(4,8,-1),hidden1.view(4,8,-1)))
        # print("++++++++++++++++++++++++++")
        # print(lstm_out.shape)   #[32,100,768]
        # print(hidden_last.shape)   #[4, 32, 384]
        # print(cn_last.shape)    #[4, 32, 384]

        # 修改 双向的需要单独处理
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

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        if (USE_CUDA):
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
            print(self.n_layers * number, batch_size, self.hidden_dim)
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )

        return hidden

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
loss_fn=nn.CrossEntropyLoss()

#####指定多路gpu训练
# model=bert_lstm()
# model = nn.DataParallel(model)
# model = model.cuda()
# print(model)
# optim = AdamW(model.parameters(), lr=5e-5)
# model.to(device)
# model.train()
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size)
print(len(test_loader))




output_size = 2
hidden_dim = 384   #768/2
n_layers = 2
bidirectional = True  #这里为True，为双向LSTM

net = bert_lstm(hidden_dim, output_size,n_layers, bidirectional)
net=nn.DataParallel(net)
net=net.cuda()
#print(net)

# loss and optimization functions
lr = 2e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 1
# batch_size=50
print_every = 7
clip = 5  # gradient clipping

# move model to GPU, if available
if (USE_CUDA):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.module.init_hidden(batch_size)
    counter = 0

    # batch loop
    for batch in train_loader:
        counter += 1
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        if (USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        # print(h[0].shape,h[1].shape)
        net.zero_grad()
        h = torch.tensor([item.cpu().detach().numpy() for item in h]).cuda()
        # print(h[0].shape)
        output = net(inputs, h[0],h[1])
        loss = criterion(output.squeeze(), labels.long())
        loss.backward()
        optimizer.step()
        if counter%9==0:
            print("Epoch: {}/{}...".format(e + 1, epochs),
                            "Step: {}...".format(counter),
                            "Loss: {:.6f}...".format(loss.item())
                            )
        # loss stats
        # if counter % print_every == 0:
        #     net.eval()
        #     with torch.no_grad():
        #         val_h = net.init_hidden(batch_size)
        #         val_losses = []
        #         for inputs, labels in valid_loader:
        #             val_h = tuple([each.data for each in val_h])
        #
        #             if (USE_CUDA):
        #                 inputs, labels = inputs.cuda(), labels.cuda()
        #
        #             output = net(inputs, val_h)
        #             # val_loss = criterion(output.squeeze(), labels.float())
        #
        #             # val_losses.append(val_loss.item())
        #
        #     net.train()
        #     print("Epoch: {}/{}...".format(e + 1, epochs),
        #           "Step: {}...".format(counter),
        #           "Loss: {:.6f}...".format(loss.item()),
        #           "Val Loss: {:.6f}".format(np.mean(val_losses)))
test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.module.init_hidden(batch_size)
h = tuple([each.data for each in h])
# print(h[0].shape,h[1].shape)
net.zero_grad()
h = torch.tensor([item.cpu().detach().numpy() for item in h]).cuda()
# print(h[0].shape)
net.eval()
# iterate over test data
for batch in test_loader:
    inputs=batch['input_ids'].to(device)
    labels=batch['labels'].to(device)
    # h = tuple([each.data for each in h])
    if (USE_CUDA):
        inputs, labels = inputs.cuda(), labels.cuda()
    output = net(inputs, h[0],h[1])
    test_loss = criterion(output.squeeze(), labels.long())
    test_losses.append(test_loss.item())

    output = torch.nn.Softmax(dim=1)(output)
    pred = torch.max(output, 1)[1]

    # compare predictions to true label
    correct_tensor = pred.eq(labels.long().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not USE_CUDA else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

