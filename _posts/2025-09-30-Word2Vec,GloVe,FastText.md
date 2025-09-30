---
layout: post
title:  "분산 표현 기반 임베딩"
summary: "Word2Vec, GloVe, FastText 임베딩 방법과 활용 코드"
author: keonhee
date: '2025-09-30 10:00:00 +0900'
category: Deeplearning, NLP
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/Embedding_WV_GV_FT/
usemathjax: true
---
<hr class="thick-hr">

# 분산 표현 기반 임베딩
<hr class="thin-hr">

분산 표현(distributed representation)은 단어를 저차원(dense) 벡터로 매핑하면서, **단어 간 의미적/통계적 유사성**을 벡터 공간에 반영

장점 :
- 차원이 낮아 효율적인 계산
- 단어 간 의미적 관계를 벡터 공간에 반영
- 머신러닝, 딥러닝 모델에 입력으로 넣을 수 있는 형태 제공

## 1. Word2Vec
단어의 의미적 유사성을 벡터 공간에 표현하는 획기적인 방법
핵심 아이디어: 비슷한 맥락에서 사용되는 단어는 비슷한 의미를 가진다.

### 1. CBOW(Continuous Bag of Words)
   - 주변 단어들로부터 중심 단어 예측
   - 속도가 빠름
   - 작은 데이터셋에서 효과적
### 2. Skip-gram
   - 중심 단어로부터 주변단어 예측
   - 더 정확한 임베딩
   - 희귀 단어 처리에 유리

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# 문장 데이터 준비
sentences = [
    "I love natural language processing",
    "Deep learning is amazing",
    "Natural language processing uses deep learning",
    "..."
]

# 토큰화
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

# Word2Vec 모델 학습
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,  # 임베딩 차원
    window=5,         # 컨텍스트 윈도우 크기
    min_count=1,      # 최소 단어 빈도
    sg=1              # 1: Skip-gram, 0: CBOW
)

# 단어 벡터 확인
print("'language' 벡터 차원:", model.wv['language'].shape)
print("\n'language'와 유사한 단어들:")
print(model.wv.most_similar('language', topn=3))

# 단어 간 유사도
similarity = model.wv.similarity('natural', 'language')
print(f"\n'natural'과 'language'의 유사도: {similarity:.4f}")
```

## 2. FastText
Word2Vec의 개선 버전으로, 단어를 문자 n-gram으로 분해하여 학습합니다.

- **특징:**
  - **미등록 단어(OOV, Out-of-Vocabulary) 처리 가능**
  - 형태소가 풍부한 언어(한국어, 터키어 등)에서 효과적
  - 오타나 신조어에 강건함
  - 접두사/접미사 정보 활용

```python
from gensim.models import FastText

# FastText 모델 학습
fasttext_model = FastText(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,  # Skip-gram
    epochs=100
)

# 학습 데이터에 있는 단어
print("'processing' 벡터 (처음 5차원):", fasttext_model.wv['processing'][:5])

# 학습되지 않은 단어도 벡터 생성 가능!
print("\n학습되지 않은 'preprocessing' 벡터 (처음 5차원):", 
      fasttext_model.wv['preprocessing'][:5])

# 단어를 subword로 분해하여 벡터 생성
# 'preprocessing' = 'pre' + 'process' + 'ing' 등의 조합으로 이해
```

## 3. GloVe (Global Vectors for Word Representation)
Word2Vec과 유사하지만, 전역 통계 정보를 활용하는 방법입니다.

- **특징:**
  - 단어 동시 출현(co-occurrence) 행렬 활용
  - 전체 코퍼스의 통계적 정보 반영
  - Word2Vec보다 학습이 안정적

보통은 사전학습된 벡터를 다운받아서 사용함.

```python
import numpy as np

def load_glove_embeddings(file_path):
    """GloVe 임베딩 파일 로드"""
    embeddings_index = {}
    with open(file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# 사용 예시 (glove.6B.100d.txt 파일 필요)
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
print(f"로드된 단어 수: {len(glove_embeddings)}")
print(f"'computer' 벡터 (처음 5차원):", glove_embeddings['computer'][:5])
```

- 학습 원리 <br>
GloVe는 단순한 예측 모델이 아니라 행렬 분해(Matrix Factorization) 방식 활용
  1. 단어-단어 동시출현 행렬(co-occurrence matrix) X 생성
     - $$X_{ij}$$ : 단어 i와 j가 같은 문맥에서 등장한 횟수
  2. 이 행렬을 직접 분해하지 않고, 다음의 **목적 함수(loss)** 를 최소화하는 방식으로 학습
  3. 해당 문제는 최적화 문제 ➡️ gradient descent 같은 방법으로 파라미터를 업데이트

👉 따라서 Word2Vec처럼 `fit(corpus)` 같은 API로 바로 학습할 수 없고, PyTorch/TensorFlow 같은 프레임워크에서 직접 학습 루프를 구현해야 합니다.

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left(w_i^T c_j + b_i + b_j - \log X_{ij}\right)^2
$$

- $$V$$ : 전체 단어 집합(vocabulary) 크기
- $$X_{ij}$$ : 단어 i와 단어 j의 동시출현(co-occurrence) 횟수
- $$\log X_{ij}$$ : 동시출현 빈도의 로그값
- $$w_i$$ : 중심 단어(center word) i의 임베딩 벡터
- $$c_j$$ : 문맥 단어(context word) j의 임베딩 벡터
- $$b_i$$, $$b_j$$ : bias 항
- $$f(X_{ij})$$ : 가중치 함수

### GloVe 학습 과정
1. 기본 설정

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import math

# 1. 예시 Corpus 및 토큰화 (전처리 완료 가정)
corpus = [
   "i love natural language processing",
   "nlp is fun",
   "i love to code in python",
   "natural language processing is amazing",
   "..."
]

## 실제 토큰화는 이렇게 진행하지 않음. 이미 전처리 완료된 데이터이기 때문에 활용 가능함
tokens = [word for sentence in corpus for word in sentence.split()]
unique_tokens = sorted(list(set(tokens)))
word_to_ix = {word: i for i, word in enumerate(unique_tokens)}
ix_to_word = {i: word for i, word in enumerate(unique_tokens)}
VOCAB_SIZE = len(unique_tokens)


EMBEDDING_DIM = 50
WINDOW_SIZE = 2
X_max = 10  # 가중치 함수 f(x)에서 사용하는 하이퍼파라미터
alpha = 0.75 # 가중치 함수 f(x)에서 사용하는 하이퍼파라미터

print(f"단어 사전 크기: {VOCAB_SIZE}")
```

2. 동시 등창 행렬(co-occurence matrix) 구성

```python
# 2. 동시 등장 행렬 X 생성
co_occurrence_matrix = defaultdict(lambda: defaultdict(int))

for i, center_word in enumerate(tokens):
    # 중심 단어의 인덱스 범위
    start = max(0, i - WINDOW_SIZE)
    end = min(len(tokens), i + WINDOW_SIZE + 1)
    
    for j in range(start, end):
        if i != j:
            context_word = tokens[j]
            # 중심 단어와 문맥 단어의 인덱스
            center_ix = word_to_ix[center_word]
            context_ix = word_to_ix[context_word]
            
            # 동시 등장 횟수 증가 (대칭으로 처리)
            co_occurrence_matrix[center_ix][context_ix] += 1
            
# PyTorch 텐서로 변환 (학습에 사용될 데이터)
X = [] # 동시 등장 횟수 (log(X_ij))
I = [] # 중심 단어 인덱스 (i)
J = [] # 문맥 단어 인덱스 (j)

for i in range(VOCAB_SIZE):
    for j in range(VOCAB_SIZE):
        count = co_occurrence_matrix[i][j]
        if count > 0:
            X.append(count)
            I.append(i)
            J.append(j)

# 학습에 사용할 텐서
X_tensor = torch.tensor(X, dtype=torch.float)
I_tensor = torch.tensor(I, dtype=torch.long)
J_tensor = torch.tensor(J, dtype=torch.long)

print(f"학습에 사용할 동시 등장 쌍 개수: {len(X_tensor)}")
```

3. Glove모델 정의

```python
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVeModel, self).__init__()
        
        # 1. 중심 단어 임베딩 행렬 (W)
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        # 2. 문맥 단어 임베딩 행렬 (W_tilde)
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        
        # 3. 중심 단어 편향 벡터 (b)
        self.bi = nn.Embedding(vocab_size, 1)
        # 4. 문맥 단어 편향 벡터 (b_tilde)
        self.bj = nn.Embedding(vocab_size, 1)
        
        # 초기화 (GloVe는 보통 랜덤하게 초기화)
        self.init_weights()

    def init_weights(self):
        # Glorot/Xavier 초기화 (균일 분포)
        bound = 1.0 / math.sqrt(EMBEDDING_DIM)
        nn.init.uniform_(self.wi.weight.data, -bound, bound)
        nn.init.uniform_(self.wj.weight.data, -bound, bound)
        
    def forward(self, i_indices, j_indices):
        # 중심 단어/문맥 단어 인덱스를 받아 임베딩 추출
        wi = self.wi(i_indices)
        wj = self.wj(j_indices)
        bi = self.bi(i_indices).squeeze()
        bj = self.bj(j_indices).squeeze()
        
        # 내적 계산: w_i^T * w_j
        dot_product = (wi * wj).sum(1)
        
        # 최종 계산 값: w_i^T * w_j + b_i + b_j
        prediction = dot_product + bi + bj
        
        return prediction
```

4. Glove 학습

```python
# 가중치 함수 f(x) 정의
def weighting_function(x, x_max=X_max, alpha=alpha):
    # x/x_max의 alpha 승, 단 x가 x_max보다 크면 1
    return torch.min(torch.ones_like(x), (x / x_max) ** alpha)

# 모델 인스턴스화
model = GloVeModel(VOCAB_SIZE, EMBEDDING_DIM)
optimizer = optim.Adagrad(model.parameters(), lr=0.05) # GloVe는 보통 Adagrad를 사용

num_epochs = 100
log_X = torch.log(X_tensor) # 미리 log(X_ij) 계산

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    
    # 1. 모델 예측 (w_i^T * w_j + b_i + b_j)
    predictions = model(I_tensor, J_tensor)
    
    # 2. 손실 항 계산 (예측 값 - log(X_ij))
    loss_term = predictions - log_X
    
    # 3. 가중치 함수 계산 (f(X_ij))
    weights = weighting_function(X_tensor)
    
    # 4. 최종 GloVe 손실 계산 (Weighted Squared Error)
    loss = (weights * (loss_term ** 2)).mean()
    
    # 5. 역전파 및 최적화
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")# 가중치 함수 f(x) 정의
def weighting_function(x, x_max=X_max, alpha=alpha):
    # x/x_max의 alpha 승, 단 x가 x_max보다 크면 1
    return torch.min(torch.ones_like(x), (x / x_max) ** alpha)

# 모델 인스턴스화
model = GloVeModel(VOCAB_SIZE, EMBEDDING_DIM)
optimizer = optim.Adagrad(model.parameters(), lr=0.05) # GloVe는 보통 Adagrad를 사용

num_epochs = 100
log_X = torch.log(X_tensor) # 미리 log(X_ij) 계산

for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    
    # 1. 모델 예측 (w_i^T * w_j + b_i + b_j)
    predictions = model(I_tensor, J_tensor)
    
    # 2. 손실 항 계산 (예측 값 - log(X_ij))
    loss_term = predictions - log_X
    
    # 3. 가중치 함수 계산 (f(X_ij))
    weights = weighting_function(X_tensor)
    
    # 4. 최종 GloVe 손실 계산 (Weighted Squared Error)
    loss = (weights * (loss_term ** 2)).mean()
    
    # 5. 역전파 및 최적화
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
```

5. 모델 활용: 최종 임베딩 추출

```python
# 최종 임베딩 행렬: (W + W_tilde) / 2
final_embeddings_tensor = (model.wi.weight.data + model.wj.weight.data) / 2

print("\n--- 학습 완료 후 최종 임베딩 확인 ---")

# 특정 단어의 임베딩 확인
word = "natural"
word_index = word_to_ix[word]

# Numpy로 변환하여 출력
embedding_vector = final_embeddings_tensor[word_index].cpu().numpy()

print(f"단어 '{word}'의 최종 임베딩 벡터 (처음 5차원):")
print(embedding_vector[:5])

# 임베딩 유사성 확인 (예시)
word1 = "i"
word2 = "love"

# 단어 인덱스
idx1 = word_to_ix[word1]
idx2 = word_to_ix[word2]

# 임베딩 벡터
vec1 = final_embeddings_tensor[idx1]
vec2 = final_embeddings_tensor[idx2]

# 코사인 유사도 계산
cosine_similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))

print(f"\n'{word1}'와 '{word2}'의 코사인 유사도: {cosine_similarity.item():.4f}")
```