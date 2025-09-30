---
layout: post
title:  "자연어 분류 - BiLSTM"
summary: "BiLSTM을 활용하여 텍스트 분류하기"
author: keonhee
date: '2025-09-30 13:00:00 +0900'
category: Deeplearning, NLP
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/text_classification/
usemathjax: true
---
# PyTorch Embedding-RNN 자연어 분류 가이드

## 개요

RNN 구조 모델은 순서의 정보를 활용하기 때문에 단어의 순서가 존재하는 자연어 분석에 활용할 수 있습니다. 텍스트 데이터를 RNN으로 학습할 때는 토큰화가 이루어지고 해당 토큰이 각 Time-Step이 됩니다. 이때 실제 RNN 층에 통과되기 이전에 Embedding을 활용하여 임베딩 하여 RNN에 입력됩니다.

## 1. 파이토치 Embedding 레이어

### 1.1 개념

파이토치는 정수 인덱스(토큰 ID)를 입력받아 해당 임베딩 벡터를 반환하는 간단한 룩업 테이블(Lookup Table) 형태의 모듈인 `nn.Embedding` 레이어를 제공합니다.

**특징:**
- 내부적으로 (num_embeddings, embedding_dim) 형태의 가중치 행렬을 보유
- 인덱스를 이용해 해당 행의 벡터를 즉시 조회(lookup)하여 반환
- 학습 과정에서 역전파를 통해 이 행렬이 업데이트되어 각 단어 임베딩이 의미 있는 벡터로 학습됨

### 1.2 주요 파라미터

- `num_embeddings`: 토큰의 개수(단어 사전 크기)
- `embedding_dim`: 임베딩 벡터의 길이(벡터 차원수)

### 1.3 기본 사용 예제

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(100, 32)
input = torch.tensor([1, 2, 3, 4])  # 임의의 토큰
output = embedding(input)

print(f'임베딩 행렬 형상: {embedding.weight.shape}')  # torch.Size([100, 32])
print(f'임베딩 결과 형상: {output.shape}')  # torch.Size([4, 32])
```

## 2. Bi-LSTM 감정 분류 모델

### 2.1 모델 구조

감정 분석을 위한 순환 신경망은 다음과 같이 구성됩니다:

1. **임베딩 레이어**: 텍스트 데이터(단어)를 고차원의 실수 벡터로 변환
2. **Bi-LSTM 레이어**: 입력된 임베딩 벡터를 시간 축을 따라 처리하여 시퀀스 데이터의 문맥 정보를 학습
3. **출력 레이어**: 양방향 LSTM의 모든 시퀀스 결과를 평균 풀링(Pooling)하여 차원 축소 후 분류

### 2.2 모델 구현

```python
import torch
import torch.nn as nn

class EmdBiLSTM(nn.Module):
    def __init__(self,
                 vocab_size, 
                 embedding_dim, 
                 hidden_size=128,
                 num_layers=4, 
                 num_classes=2):
        super(EmdBiLSTM, self).__init__()
        
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bi-LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 임베딩
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Bi-LSTM
        lstm_out, (h, c) = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]
        
        # Global Average Pooling
        pooled = torch.mean(lstm_out, dim=1)
        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        
        # 분류 레이어
        dense = self.fc1(pooled)
        out = self.fc2(dense)
        
        return out
```

## 3. SentencePiece 토큰화

### 3.1 모델 로드 및 토큰화

```python
import sentencepiece as spm

# SentencePiece 모델 로드
sp = spm.SentencePieceProcessor(model_file='spm_krsent.model')

# 문자열 토큰화
str_tokens = sp.encode(text, out_type=str)

# 인코딩 숫자 토큰화
ind_tokens = sp.encode(text, out_type=int)
```

### 3.2 패딩과 최대 길이 설정

딥러닝 모델은 입력 형상이 항상 고정되어야 하므로 다음과 같은 방식으로 처리합니다:

- **최대 길이를 넘어가는 경우**: 넘어가는 토큰을 잘라냄
- **최대 길이에 못미치는 경우**: 부족한 토큰을 패딩 토큰(0)으로 채움

```python
def tokenize_with_spm(text, sp, max_len):
    token_ids = sp.encode(text, out_type=int)
    
    if len(token_ids) < max_len:
        # 0 패딩 추가
        token_ids += [0] * (max_len - len(token_ids))
    else:
        # 초과시 자름
        token_ids = token_ids[:max_len]
    
    return token_ids
```

## 4. PyTorch 데이터세트 구성

### 4.1 커스텀 데이터세트 클래스

```python
from torch.utils.data import Dataset
import numpy as np

class SPDataSet(Dataset):
    def __init__(self, df, sp, max_len):
        self.max_len = max_len
        self.df = df
        self.sp = sp
        self.class_name = {'E1': 0, 'E6': 1, 'E3': 2, 'E5': 3, 'E2': 4, 'E4': 5}

    def zero_pad(self, tok):
        if len(tok) >= self.max_len:
            return tok[:self.max_len]
        else:
            padding = np.zeros(self.max_len)
            padding[:len(tok)] = tok
            return padding

    def __getitem__(self, i):
        inp = str(self.df.iloc[i]['text'])
        tar = self.df.iloc[i]['label']
        
        # 라벨 인코딩
        tar = self.class_name[tar]
        
        # 문장 인코딩 (시작토큰과 끝 토큰 추가)
        inp = [self.sp.bos_id()] + self.sp.encode_as_ids(inp) + [self.sp.eos_id()]
        
        # 패딩
        inp = self.zero_pad(inp)
        
        return torch.Tensor(inp), tar

    def __len__(self):
        return len(self.df)
```

### 4.2 데이터로더 생성

```python
from torch.utils.data import DataLoader, random_split

sp = spm.SentencePieceProcessor(model_file='spm_krsent.model')
dataset = SPDataSet(sent_df, sp, max_len=60)

# 데이터세트 분할
generator1 = torch.Generator().manual_seed(42)
test_dataset, train_dataset = random_split(dataset, [0.2, 0.8], generator=generator1)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## 5. 모델 학습

### 5.1 하이퍼파라미터 설정

```python
# 하이퍼파라미터 설정
embedding_dim = 128
max_len = 60
hidden_size = 256
num_layers = 3
num_classes = 6
learning_rate = 1e-4
num_epochs = 20
batch_size = 64

vocab_size = sp.get_piece_size()
```

### 5.2 모델 초기화 및 학습

```python
import torch.optim as optim

# 모델 초기화
model = EmdBiLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes
)

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 히스토리 저장
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for input, labels in train_loader:
        input_ids = input.long().to(device)
        labels = labels.to(device)
        
        # 순전파
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_loss)
    
    # 검증
    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0

    with torch.no_grad():
        for input, labels in val_loader:
            input_ids = input.long().to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            total_val_loss += loss.item()
            acc = (outputs.argmax(dim=1) == labels).sum() / len(labels)
            total_val_acc += acc

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)
    
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc.cpu().numpy())
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Loss: {avg_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {avg_val_acc:.4f}")
```

## 6. Word2Vec 임베딩 적용

### 6.1 Word2Vec 모델 학습

```python
from gensim.models import Word2Vec

# 문장을 토큰으로 변환
sentences = sent_df['text'].apply(lambda x: sp.encode(str(x), out_type=str))
sentences = sentences.tolist()

# Word2Vec 모델 학습
emd_dim = 128
model = Word2Vec(
    sentences=sentences,
    vector_size=emd_dim,
    window=3,
    min_count=2,
    workers=4,
    sg=1,  # Skip-gram
    epochs=10
)

model.save("word2vec.model")
```

### 6.2 임베딩 행렬 생성

```python
# 임베딩 크기와 토큰 개수
embedding_dim = model.vector_size
sp_vocab_size = sp.get_piece_size()

# 임베딩 텐서 초기화
embedding_matrix = torch.zeros((sp_vocab_size, embedding_dim), dtype=torch.float)

# 토큰을 Word2Vec의 임베딩 벡터에 매핑
for idx in range(sp_vocab_size):
    token = sp.id_to_piece(idx)
    
    if token in model.wv.key_to_index:
        embedding_matrix[idx] = torch.tensor(model.wv[token])
```

### 6.3 사전 학습된 임베딩 적용

```python
class W2VBiLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 embedding_matrix=None,
                 hidden_size=128,
                 num_layers=4,
                 num_classes=2):
        super(W2VBiLSTM, self).__init__()

        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 사전 학습된 임베딩 적용
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)

        # Bi-LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 출력 레이어
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        pooled = torch.mean(out, dim=1)  # Global Average Pooling
        out = self.fc(pooled)
        return out
```

## 7. 학습 결과 시각화

```python
import matplotlib.pyplot as plt

loss = history['train_loss']
val_loss = history['val_loss']
val_acc = history['val_acc']

eps = range(len(val_loss))

fig = plt.figure(figsize=(10, 5))

# 손실 그래프
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(eps, val_loss, label='val_loss')
ax1.plot(eps, loss, label='train_loss')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

# 정확도 그래프
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(eps, val_acc, label='val_acc')
ax2.legend()
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')

plt.show()
```

## 8. 주요 포인트

### 8.1 임베딩 레이어의 역할
- 정수 인덱스를 밀집 벡터로 변환
- 학습을 통해 의미론적으로 유사한 단어들이 가까운 벡터 공간에 위치

### 8.2 Bi-LSTM의 장점
- 양방향 처리로 과거와 미래 문맥 모두 활용
- 시퀀스 데이터의 장기 의존성 학습 가능

### 8.3 사전 학습된 임베딩 사용
- Word2Vec 등으로 사전 학습된 임베딩 사용 시 더 나은 초기화 가능
- 학습 데이터가 적을 때 특히 효과적

### 8.4 패딩 처리
- 고정된 입력 크기를 위해 필수
- 짧은 시퀀스는 0으로 채우고, 긴 시퀀스는 자름
