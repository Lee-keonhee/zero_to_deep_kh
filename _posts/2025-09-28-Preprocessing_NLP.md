---
layout: post
title: 자연어 전처리(Preprocessing_NL)
summary: 자연어 전처리 종류 및 과정
author: keonhee
date: 2025-09-28 09:00:00 +0900
category: Deeplearning, NLP
keywords: 딥러닝
permalink: /blog/Preprocessing_NL/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
<hr class="thick-hr">

# ✍️ 자연어 전처리 기초: 토큰화, 정제, 정규화
<hr class="thin-hr">

## 자연어 전처리란?
<hr class="thin-hr">

텍스트 데이터를 인공지능 모델이 효과적으로 학습할 수 있도록 가공하는 과정을 **자연어 전처리(NLP Preprocessing)** 라고 합니다. 이 과정은 크게 **정제(Cleaning)**, **토큰화(Tokenization)**, **정규화(Normalization)** 세 단계로 나뉩니다.<br>

### 1. 정제 (Cleaning)
데이터의 품질을 높이고 분석의 정확도를 개선하기 위해 불필요하거나 방해가 되는 요소를 제거하는 초기 단계입니다.

#### A. 불필요한 문장 부호 및 기호 제거 (영어/ 한글 공통)
분석에 직접적인 의미를 제공하지 않는 문장 부호나 특수 기호 등을 제거합니다.

- 원문 예시: Oh, Hi hello. Nice to meet you!!!
- 수정 후: Oh Hi hello Nice to meet you (특수문자, 2칸이상의 공백, tab, 줄 바꿈 등 제거)

```python
import re

text = 'Oh, Hi hello. Nice to meet you!!!'
rtext = re.sub(r'[^\w\s]+', '', text)
rtext = re.sub(r'[\n\t]+', '', rtext)
rtext = re.sub(r'\s+', ' ', rtext)
```

#### B. 대문자 → 소문자 변환 (Lowercasing) (영어 단독)
같은 단어지만 대소문자 표기가 다를 경우 서로 다른 단어로 인식되는 것을 방지하기 위해 모두 소문자로 통일합니다.

- 원문 예시: Oh, Hi hello. Nice to meet you.
- 수정 후: oh, hi hello. nice to meet you.

```python
text =  'Oh, Hi hello. Nice to meet you.'
rtext = str.lower(text)
```

#### C. 불필요한 단어 제거 (Stopword Removal)
분석에 기여하지 않는 단어(감탄사, 관사, 전치사 등)나 지나치게 중복되는 단어를 제거하여 노이즈를 줄입니다.

- 분석 기여도가 낮은 단어 제거:<br>
    분석에 기여하지 않는 단어(감탄사 등)를 제거합니다.
  - 원문: Oh, Hi hello. Nice to meet you.
  - 수정 후: Hi hello. Nice to meet you. (감탄사 'Oh,' 제거)<br>

- 중복 단어 제거:
  - 원문: hello hello nice to meet you
  - 수정 후: hello nice to meet you



### 2. 토큰화 (Tokenization)
주어진 텍스트를 모델이 처리할 수 있는 가장 작은 의미 단위인 **토큰(Token)** 으로 나누는 과정입니다.

- 단어 토큰화 (Word Tokenization)의 특징:

    - "happily-ever-after"와 같이 하이픈(-)으로 연결된 단어는 그대로 하나의 토큰으로 처리될 수 있습니다.
    - "ending,"과 같이 단어와 구두점이 붙어 있는 경우, $\text{NLTK}$와 같은 표준 토크나이저는 단어와 콤마를 분리하여 두 개의 토큰 (ending, ,)으로 나눕니다.

Python (NLTK)을 활용한 토큰화 및 정제 예시
다음 코드는 NLTK 라이브러리를 사용해 토큰화, 불용어(Stopwords) 제거, 그리고 알파벳이 아닌 문자를 필터링하는 과정을 보여줍니다.
#### 영어

```python
"""띄어쓰기가 잘되어있는 언어(영어, 프랑스어 등등)"""
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# 텍스트 예시 (영화 리뷰 코멘트)
text = """ After reading the comments for this movie, I am not sure whether I should be angry, sad or sickened. Seeing comments typical of people who know absolutely nothing about the military or who base everything they think they know on movies like this makes me wonder about the state of intellectual stimulation in the world. At the time I type this, the number of people in the US military: 1.4 million on Active Duty with another almost 900,000 in the Guard and Reserves for a total of roughly 2.3 million. The number of people indicted for abuses at Abu-Gharib: currently less than 20. Even if you indict every single military member that ever stepped into Abu-Gharib, you would not come close to making that a whole number. The flaws in this movie would take years to cover. """

# 1. 단어 토큰화
words = word_tokenize(text)

# 2. 불용어 및 비알파벳 문자 제거 (정제)
stop_words = set(stopwords.words('english'))
words_no_stopwords = [
    word for word in words 
    if word.lower() not in stop_words and word.isalpha()
]

# 3. 빈도수 계산
word_counts = Counter(words_no_stopwords)

print("--- 토큰화 및 불용어 제거 결과 (상위 20개) ---")
print(words_no_stopwords[:20]) 
print("\n--- 상위 5개 단어 빈도수 ---")
print(word_counts.most_common(5))
```
#### 한국어

```python
"""한국어, 중국어, 일본어 등 동양권 언어"""
import sentencepiece as spm
import pandas as pd
import re

df = pd.read_csv('./nreview_mask.csv')

with open('./nreview_mask.txt', 'w', encoding='utf-8') as af:
    for text in df['text']:
        text = str(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[\n\t]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        try:
            af.write(text+'\n')
        except:
            pass
        
import os

os.makedirs('./model', exist_ok=True)

spm.SentencePieceTrainer.train(input='./nreview_mask.txt',
                               model_prefix='./model/spm',
                               vocab_size=2000
                               )

sp = spm.SentencePieceProcessor(model_file='./model/korean_sp.model')
df = pd.read_csv('./nreview_mask.csv')

text = df.loc[3,'text']

print(sp.encode(text, out_type=str))
print(sp.encode(text, out_type=int))
```

### 3. 정규화 (Normalization)
서로 다른 형태를 가진 단어들을 그 의미를 유지하면서 통일된 하나의 기본형으로 만드는 과정입니다.

#### A. 어간 추출 (Stemming)
- 정의: 단어의 접미사나 어미를 기계적인 규칙에 따라 잘라내어 **어간(Stem)**을 추출하는 방식입니다.

- 특징: 단순하고 빠르지만, 결과가 실제 존재하는 단어 형태가 아닐 수 있습니다. (예: amusing → amus)

- Stemming 결과 예시 (일부):<br>
    ```['see', 'comment', 'typic', 'of','peopl', 'who', 'absolut', 'know', 'noth', 'about', 'the', 'militari', 'or', 'who', 'base', 'everyth', 'on', 'movi', 'like', 'thi', 'make', 'me', 'wonder', '.']```<br>
    (Stemming은 military를 militari로, people을 peopl로 변환하여 실제 단어가 아닌 형태로 만들 수 있습니다.)

##### A-1. 포터 스테머 알고리즘을 통한 어간 추출
```python
from nltk.stem import PorterStemmer
from nltk import word_tokenize

porter_stemmer = PorterStermmer()
text = 'You are so lovely. I am loving you now.'
porter_stemmed_words = []

# 단어 토큰화
tokenized_words = word_tokenize(text)
for word in tokenized_words:
    stem = porter_stemmer(word)
    porter_stemmed_words.append(stem)
```
##### A-2. 랭커스터 스테머 알고리즘을 통한 어간 추출
```python
from nltk.stem import LancasterStemmer
from nltk import word_tokenize

lancaster_stemmer = LancasterStemmer()
text = "You are so lovely. I am loving you now."
lancaster_stemmed_words = []
tokenized_words = word_tokenize(text)

# 랭커스터 스테머의 어간 추출
for word in tokenized_words:
    stem = lancaster_stemmer.stem(word)
    lancaster_stemmed_words.append(stem)

```

#### B.표제어 추출 (Lemmatization)
정의: 단어의 품사 정보와 사전 지식을 사용하여 단어의 **기본형 단어(Lemma)** 를 찾는 방식입니다.

특징: 정확도가 높으며, 결과가 항상 실제 사전에 존재하는 단어 형태입니다. (예: ```am```, ```are```, ```is``` → ```be```)

```LLM``` 이나 복잡한 ```NLP``` 모델에서는 일반적으로 ```Lemmatization```이나 ```Stemming``` 대신, ***서브워드 토큰화(```Subword Tokenization```)*** 를 통해 단어의 의미적 유사성을 벡터 공간에 담아 정규화 문제를 간접적으로 해결합니다.