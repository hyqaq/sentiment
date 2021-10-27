# from transformers import pipeline,AutoTokenizer, AutoModel
# classifier = pipeline('sentiment-analysis')
# a=classifier('We are very happy to introduce pipeline to the transformers repository.')
# print(a)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")
# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)
# print(inputs)
# from datasets import load_dataset
# raw_dataset=load_dataset("imdb")
import torch.cuda
import datasets
from datasets import load_dataset
import random
raw_dataset = load_dataset("imdb")
#查看数据集，内容
print(raw_dataset)
# for i in range(10):
#     print(raw_dataset["train"][random.randint(0,25000)])
print(raw_dataset["train"][:5])
print(raw_dataset["train"].features)
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")

#重写函数，其中examples["text"]中的“text”时看数据集的feature
def tokenize_function(examples):
    return tokenizer(examples["text"],padding="max_length",truncation=True)


# token_datasets=raw_dataset.map(tokenize_function,batched=True,batch_size=16)
token_datasets=raw_dataset.map(tokenize_function,batched=True)
# small_train_dataset=token_datasets["train"].shuffle(seed=5).select(range(1000))
# small_test_dataset=token_datasets["test"].shuffle(seed=5).select(range(1000))
small_train_dataset=token_datasets["train"]
small_test_dataset=token_datasets["test"]

from transformers import AutoModelForSequenceClassification

model=AutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=2)

from transformers import TrainingArguments

#这里设置训练器的参数，这里其实啥也没设置，都是默认的。
training_args=TrainingArguments("test_trainer")

from transformers import Trainer
trainer=Trainer(model=model,args=training_args,train_dataset=small_train_dataset,eval_dataset=small_test_dataset)
# if hasattr(torch.cuda,'empty_cache'):
#     torch.cuda.empty_cache()
trainer.train()
# if hasattr(torch.cuda,'empty_cache'):
#     torch.cuda.empty_cache()

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics,
)
print(trainer.evaluate())
