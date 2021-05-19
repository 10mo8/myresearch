#BERTを用いて文章の要約を行うシステムです
#CLSトークンの出力を利用しニューラルネットワークの重みを学習します。
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from transformers import BertModel
from transformers import AutoTokenizer
model_path = "./data/model.pth"

# データの読み込み
df = pd.read_csv("answer_qan.csv", encoding="utf-8", header=None, names=['sentence', 'label'])

# データの抽出
sentences = df.sentence.values
labels = df.label.values

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

#BERTは512単語まで扱える
MAX_LENGTH = 512
def bert_tokenizer(text):
    return tokenizer.encode(text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")[0]

TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=bert_tokenizer, use_vocab=False, lower=False,
                            include_lengths=True, batch_first=True, fix_length=MAX_LENGTH, pad_token=0)
LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

train_data, test_data = torchtext.legacy.data.TabularDataset.splits(
    path="./", train="train.csv", test="test.csv", format="csv", fields=[("Text", TEXT), ("Label", LABEL)]
)

#BERTではミニバッチサイズは16or32
BATCH_SIZE =32
train_iter, test_iter = torchtext.legacy.data.Iterator.splits((train_data, test_data), batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        #日本語学習済みモデルをロードする
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',
                                              output_attentions=True,
                                              output_hidden_states=True)
        
        #BERTの隠れ層次元は768だが、最終4層を利用する
        self.linear = nn.Linear(768, 2)
        
        #重み初期化
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    #clsトークンのベクトルを取得する
    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, 768)
    
    def forward(self, input_ids):
        #順伝播の出力結果は辞書形式なので、必要なkeyを指定する
        output = self.bert(input_ids)
        attentions = output["attentions"]
        hidden_states = output["hidden_states"]
        
        #最終層からslcトークンのベクトル取得
        vec1 = self._get_cls_vec(hidden_states[-1])
        
        #全結合層でクラス分類用に次元変換
        out = self.linear(vec1)
        
        return F.log_softmax(out, dim=0), attentions
    
classifier = BertClassifier()

# まずは全部OFF
for param in classifier.parameters():
    param.requires_grad = False

# BERTの最終層をON
for param in classifier.bert.encoder.layer[-1].parameters():
    param.requires_grad = True

# クラス分類のところもON
for param in classifier.linear.parameters():
    param.requires_grad = True

# 事前学習済の箇所は学習率小さめ、最後の全結合層は大きめにする。
optimizer = optim.Adam([
    {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': classifier.linear.parameters(), 'lr': 1e-4}
])

# 損失関数の設定
loss_function = nn.CrossEntropyLoss()

# GPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ネットワークをGPUへ送る
classifier.to(device)
losses = []
#モデルの読み込み
#classifier.load_state_dict(torch.load(model_path))

# エポック数は50で
for epoch in range(50):
    all_loss = 0

    for idx, batch in enumerate(train_iter):
        classifier.zero_grad()

        input_ids = batch.Text[0]
        label_ids = batch.Label
        print(input_ids)
        print(input_ids.shape)

        out, _ = classifier(input_ids)
       
        batch_loss = loss_function(out, label_ids)
        batch_loss.backward()

        optimizer.step()

        all_loss += batch_loss.item()

    print("epoch", epoch, "\t" , "loss", all_loss)

#モデルの保存
#torch.save(classifier.state_dict(), model_path)
answer = []
prediction = []

with torch.no_grad():
    for batch in test_iter:

        text_tensor = batch.Text[0]
        label_tensor = batch.Label
        
        print(text_tensor)
        print(text_tensor.shape)
        score, _ = classifier(text_tensor)
        print(score)
        _, pred = torch.max(score, 1)
        print(pred)

        prediction += list(pred.numpy())
        answer += list(label_tensor.numpy())
        print(label_tensor)
        print(pred)

print(classification_report(prediction, answer))
