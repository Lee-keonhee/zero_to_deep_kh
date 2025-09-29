---
layout: post
title:  "자연어 임베딩(Embedding_NL)"
summary: "자연어 인코딩 및 임베딩 과정"
author: keonhee
date: '2025-09-29 15:00:00 +0900'
category: Deeplearning, NLP
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/Embedding_NL/
usemathjax: true
---
<hr class="thick-hr">

# ✍️ 자연어 임베딩
<hr class="thin-hr">

## 자연어 임베딩이란?
<hr class="thin-hr">

컴퓨터는 텍스트를 직접 이해할 수 없기 때문에, 단어나 문장을 숫자로 된 **벡터(Vector)** 로 변환해야 합니다. 이 과정을 **임베딩(Embedding)** 또는 **벡터화(Vectorization)** 라고 합니다. 임베딩은 단어의 의미적 유사성을 수치적으로 표현하여, 비슷한 의미를 가진 단어들이 벡터 공간상에서 가까이 위치하도록 만듭니다.

### 1. 기본 벡터화 방법
<hr class="thin-hr">

#### A. One-Hot Encoding
가장 단순한 벡터화 방식으로, 각 단어를 고유한 인덱스로 표현.<br>
특징:
- 어휘 크기만큼의 차원을 가진 벡터 생성
- 해당 단어의 인덱스 위치만 1, 나머지는 0
- 단어 간 유사도를 전혀 표현하지 못함

```python
# 어휘: ['cat', 'dog','bird']
cat = [1,0,0]
dog = [0,1,0]
bird = [0,0,1]
```

문제점 : 
- 어휘 크기가 커질수록 벡터 차원이 급격히 증가(희소성 문제: 벡터 차원은 엄청 큰데 데이터는 적은 문제)
- 어휘끼리의 유사도를 표현할 수 없음
- 비효율적 메모리 사용

#### B. Bag of Words(BoW)
문서를 단어의 출현 빈도로 표현하는 방법.
특징 :
- 단어의 순서 정보 무시
- 각 문서를 어휘 크기만큼의 벡터로 표현
- 해당 단어의 출현 횟수를 벡터값으로 사용

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love natural language processing",
    "I love deep learning",
    "자연어 처리는 대단하다"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

# 표로 시각화
import pandas as pd
df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df.head()
```

```python
#출력
|    |deep  | language | learning | love | natural | processing | 대단하다 | 자연어  | 처리는 |
|----|------|----------|----------|------|---------|------------|---------|--------|-------|
|0   |  0   |     1    |     0    |  1   |    1    |      1     |    0    |    0   |   0   |
|1   |   1  |     0    |    1     |  1   |    0    |      0     |    0    |    0   |   0   |
|2   |  0   |     0    |    0     |  0   |    0    |      0     |    1    |    1   |   1   |
```

#### C. TF-IDF(Term Frequeny-Inverse Document Frequency)
 단순 빈도 수가 아닌, 문서 내 중요도를 고려한 가중치를 부여합니다.
수식: 

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

$$\text{TF}(t, d) = \frac{\text{문서 } d \text{에서 단어 } t \text{의 출현 횟수}}{\text{문서 } d \text{의 총 단어 수}}$$

$$\text{IDF}(t) = \log\left(\frac{\text{전체 문서 수}}{\text{단어 } t \text{를 포함하는 문서 수}}\right)$$

특징 :
- 자주 등장하지만 많은 문서에 공통적으로 많이 나타나는 단어(```the```, ```is```)의 가중치는 낮춤
- 특정 문서에만 많이 등장하는 단어에 가중치를 높힘
- 문서 분류, 검색 엔진 등에 효과적

```python
import os
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love natural language processing",
    "I love deep learning",
    "자연어 처리는 대단하다"
    ]

vectorizer = TfidfVectorizer()
Tf_matrix = vectorizer.fit_transform(corpus)

# print(Tf_matrix)
df = pd.DataFrame(Tf_matrix.toarray(),columns=vectorizer.get_feature_names_out())
print(df.head())
```

```python
#출력
|    |deep  | language | learning | love | natural | processing | 대단하다 | 자연어  | 처리는 |
|----|------|----------|----------|------|---------|------------|---------|--------|-------|
|0   |  0   |     1    |     0    |  1   |    1    |      1     |    0    |    0   |   0   |
|1   |   1  |     0    |    1     |  1   |    0    |      0     |    0    |    0   |   0   |
|2   |  0   |     0    |    0     |  0   |    0    |      0     |    1    |    1   |   1   |
```
