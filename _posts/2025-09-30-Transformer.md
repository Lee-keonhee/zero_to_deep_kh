---
layout: post
title:  "Transformer 분석"
summary: "Transformer 모델 분석 및 정리"
author: keonhee
date: '2025-09-30 10:00:00 +0900'
category: Deeplearning, NLP
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/Transformer/
usemathjax: true
---

<hr class="thick-hr">

# 🧠 LLM의 핵심: 트랜스포머 아키텍처 분석

거대 언어 모델(LLM)은 **트랜스포머(Transformer)** 구조를 기반으로 하며, 이 구조의 혁신은 순환 신경망(RNN)이 가진 느린 속도와 장기 의존성 문제를 해결한 **어텐션 메커니즘**에 있습니다.

<hr class="thin-hr">

## 1단계: 트랜스포머의 심장 — 어텐션 메커니즘

트랜스포머는 문장의 모든 단어를 **병렬**로 처리하며, 각 단어가 문맥을 파악하기 위해 다른 단어에 **집중(Attention)**하는 방식을 사용합니다.

### 1.1. 셀프 어텐션 (Self-Attention)의 작동 원리

어텐션은 모든 단어의 임베딩 벡터에서 파생된 세 가지 벡터를 사용합니다.

| 요소 | 역할 | 핵심 기능 |
|------|------|-----------|
| **Q (Query)** | 현재 단어가 '어떤 단어에 집중해야 할까?'라고 묻는 | **주체** |
| **K (Key)** | 문장 내 다른 단어들이 '나와 얼마나 관련 있는가?'라고 답하는 | **열쇠** |
| **V (Value)** | 다른 단어들이 가진 | **실제 의미 정보** |

#### Scaled Dot-Product Attention 공식

Q와 K를 내적하여 유사도 점수를 구하고, Softmax를 적용하여 **어텐션 가중치(합=1)**로 변환한 뒤 V와 행렬 곱셈(`@`)으로 가중합을 계산합니다.

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**수식 설명:**
- $QK^T$: Query와 Key의 내적으로 유사도 계산
- $\sqrt{d_k}$: 스케일링 팩터 (벡터 차원의 제곱근)
- $\text{Softmax}$: 유사도 점수를 확률 분포로 변환
- 최종적으로 V와 곱하여 가중 평균된 문맥 벡터 생성

### 1.2. 다중 헤드 어텐션 (Multi-Head Attention, MHA)

**목표:** 단 한 번의 어텐션 대신, 입력 벡터를 여러 개의 작은 덩어리(헤드)로 나누어 **병렬적**으로 어텐션을 수행합니다.

**이점:** 하나의 헤드가 **주어-서술어** 관계에 집중한다면, 다른 헤드는 **명사-대명사** 관계에 집중하는 등, **다양한 관점**에서 문맥 관계를 동시에 포착하여 언어 이해 능력을 극대화합니다.

**작동 방식:**
1. 입력을 h개의 헤드로 분할 ($d_{model} / h$ 차원씩)
2. 각 헤드에서 독립적으로 어텐션 계산
3. 모든 헤드의 출력을 연결(Concatenate)
4. 선형 변환으로 최종 출력 생성

<hr class="thin-hr">

## 2단계: 구조 안정화 및 정보 주입

트랜스포머 블록은 깊은 네트워크를 안정화하고 순서 정보를 제공하는 추가적인 요소를 포함합니다.

### 2.1. 잔차 연결 (Residual Connection, Add)

**원리:** 입력 $\mathbf{x}$를 서브 레이어의 출력에 그대로 더해 다음 층으로 전달하는 **지름길**을 만듭니다.

$$
\text{Output} = \mathbf{x} + \text{Sublayer}(\mathbf{x})
$$

**역할:** 모델이 깊어질 때 발생하는 **기울기 소실(Vanishing Gradient)** 문제를 방지하여 학습을 안정화시키는 데 필수적입니다.

**장점:**
- 깊은 네트워크에서도 기울기가 잘 전파됨
- 항등 함수(Identity Function)를 학습하기 쉬움
- 각 레이어가 입력에 대한 "변화량"만 학습하면 됨

### 2.2. 레이어 정규화 (Layer Normalization, Norm)

**목표:** 배치(Batch) 차원이 아닌 **개별 토큰의 특징 벡터** 차원($D_{model}$)을 따라 정규화를 수행합니다.

**배치 정규화(BN) 대신 LN을 쓰는 이유:**

LLM은 모델 크기가 커서 **배치 크기가 매우 작을** 때가 많습니다. LN은 배치 크기에 **독립적**으로 작동하여 불안정한 통계를 피하고 안정적인 학습을 보장합니다.

**Layer Normalization 수식:**

$$
\text{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서:
- $\mu$: 특징 차원에 대한 평균
- $\sigma^2$: 특징 차원에 대한 분산
- $\gamma, \beta$: 학습 가능한 파라미터
- $\epsilon$: 수치 안정성을 위한 작은 상수

### 2.3. 위치 인코딩 (Positional Encoding, PE)

**목표:** 병렬 처리 때문에 사라진 단어의 **순서 정보**를 임베딩 벡터에 더해줍니다.

**사인/코사인 함수 기반의 PE 공식:**

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{aligned}
$$

여기서:
- $pos$: 토큰의 위치 (0, 1, 2, ...)
- $i$: 임베딩 차원의 인덱스
- $d_{model}$: 모델의 임베딩 차원

**사인/코사인 함수 기반의 PE 장점:**

학습 가능한 PE와 달리, 수식으로 정의되어 있기 때문에 학습 시 보지 못한 **아주 긴 문장**에도 새로운 위치 벡터를 **계산**하여 적용할 수 있는 **일반화 능력**을 갖습니다.

**추가 장점:**
- 상대적 위치 관계를 표현 가능
- 파라미터 수 증가 없음
- 임의 길이의 시퀀스에 대응 가능

<hr class="thin-hr">

## 3단계: LLM의 완성: 학습 목표 및 정렬

### 3.1. GPT 계열의 학습 목표

| 구조 | 핵심 기법 | 사전 학습 목표 |
|------|-----------|----------------|
| **디코더 전용 (GPT)** | Masked Self-Attention | 다음 단어 예측 (Next Token Prediction) |

#### Masked Self-Attention

**개념:** 어텐션 계산 시 **미래의 단어(아직 생성되지 않은)** 정보를 보지 못하도록 마스킹하여, 텍스트 생성의 **인과 관계(Causality)**를 유지합니다.

**구현 방법:**
- 어텐션 스코어 행렬에서 상삼각 부분을 $-\infty$로 설정
- Softmax 적용 시 미래 토큰에 대한 가중치가 0이 됨

**효과:** 이 단순한 '다음 단어 예측' 임무를 통해 모델은 언어의 문법, 지식, 문맥을 내재화합니다.

**학습 과정:**
1. 대규모 텍스트 코퍼스에서 문장 수집
2. 각 위치에서 이전 단어들만 보고 다음 단어 예측
3. 예측과 실제 단어의 차이를 손실로 계산
4. 역전파로 모델 파라미터 업데이트

### 3.2. 최종 정렬 (Alignment)

#### RLHF (인간 피드백 기반 강화 학습)

**정의:** 사전 학습(Pre-training)으로 언어 능력을 갖춘 LLM을 **인간의 의도, 가치, 안전 기준**에 맞게 조정하는 최종 정렬 과정입니다.

**프로세스:**
1. **지도 학습 미세 조정 (SFT)**: 고품질의 인간 작성 답변으로 모델을 미세 조정
2. **보상 모델 학습**: 인간 평가자가 여러 답변의 선호도를 매긴 데이터로 보상 모델 훈련
3. **강화 학습**: 보상 모델의 점수를 최대화하도록 LLM을 PPO 등의 강화 학습 알고리즘으로 최적화

**목표:**
- 유용성(Helpfulness): 사용자 의도에 맞는 답변 생성
- 정직성(Honesty): 사실에 기반한 정확한 정보 제공
- 무해성(Harmlessness): 윤리적, 안전한 답변 생성

<hr class="thin-hr">

## 트랜스포머 아키텍처 전체 구조

### 인코더-디코더 구조 (원본 트랜스포머)

**인코더:**
1. 입력 임베딩 + 위치 인코딩
2. Multi-Head Self-Attention
3. Add & Norm (잔차 연결 + 정규화)
4. Feed-Forward Network
5. Add & Norm
6. N개의 인코더 블록 반복

**디코더:**
1. 출력 임베딩 + 위치 인코딩
2. Masked Multi-Head Self-Attention
3. Add & Norm
4. Multi-Head Cross-Attention (인코더 출력 활용)
5. Add & Norm
6. Feed-Forward Network
7. Add & Norm
8. N개의 디코더 블록 반복

### GPT (디코더 전용)

- 디코더 구조만 사용
- Cross-Attention 제거
- Masked Self-Attention으로 자기회귀적 생성
- 사전 학습: Next Token Prediction

### BERT (인코더 전용)

- 인코더 구조만 사용
- 양방향 문맥 학습
- 사전 학습: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)

<hr class="thin-hr">

## 핵심 요약

### 트랜스포머의 혁신

1. **병렬 처리**: RNN의 순차 처리 대신 모든 토큰을 동시에 처리
2. **어텐션 메커니즘**: 문맥에 따라 중요한 정보에 집중
3. **확장성**: 모델 크기와 데이터 크기에 따라 성능 향상

### 성공의 핵심 요소

1. **Self-Attention**: 장거리 의존성을 효율적으로 포착
2. **Multi-Head Attention**: 다양한 관점의 정보 통합
3. **잔차 연결**: 깊은 네트워크의 안정적 학습
4. **Layer Normalization**: 작은 배치에서도 안정적 학습
5. **위치 인코딩**: 순서 정보 보존

### LLM의 학습 파이프라인

```
사전 학습 (Pre-training)
    ↓
지도 학습 미세 조정 (SFT)
    ↓
보상 모델 학습
    ↓
강화 학습 (RLHF)
    ↓
정렬된 LLM
```

<hr class="thick-hr">

