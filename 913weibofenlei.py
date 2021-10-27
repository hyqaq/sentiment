
from transformers import pipeline,BertForSequenceClassification
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
import csv
# classifier=pipeline('sentiment-analysis',model="techthiyanes/chinese_sentiment")

classifier=pipeline('sentiment-analysis',model="bert-base-chinese")
with open("weibo_review2.csv","r",encoding='utf-8') as f:
    reader=csv.DictReader(f)
    c=[row["评论内容"] for row in reader]
print(c)
results=classifier(c)
count=0
count2=0
count3=0
for r in results:
    if r["label"]=="star 1"or r["label"]=="star 2":
        count+=1
    elif r["label"]=="star 4"or r["label"]=="star 5":
        count2+=1
    else:
        count3+=1
negative=0
positive=0
for r in results:
    if r["label"]=="NEGATIVE":
        negative+=1
    else:
        positive+=1
print("negative: %.2f.  positive: %.2f.   "%(negative/len(results),positive/len(results)))


# for i,r in  enumerate(results):
#     if r["label"]=="star 1"or r["label"]=="star 2":
#         count+=1
#     elif r["label"]=="star 4"or r["label"]=="star 5":
#         count2+=1
#     else:
#         count3+=1

print("negative: %.2f.  positive: %.2f.   neutral:%.2f  "%(count/len(results),count2/len(results),count3/len(results)))
# tokenizer=BertTokenizerFast.from_pretrained('bert-base-chinese')
# test_encodings=tokenizer(c,truncation=True,padding=True)
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese')