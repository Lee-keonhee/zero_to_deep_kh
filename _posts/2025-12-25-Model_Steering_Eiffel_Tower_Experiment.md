---
layout: post
title: AI의 생각을 조종하는 기술 - 모델 스티어링(Model Steering)
summary: Activation Steering과 SAE를 활용한 LLM 행동 제어 - 에펠탑 벡터 실험
author: keonhee
date: 2025-12-25 09:00:00 +0900
category: AI
keywords: Model Steering, Activation Steering, SAE, Representation Engineering
permalink: /blog/model_steering_eiffel_tower/
usemathjax: true
thumbnail: /assets/img/posts/model_steering.png
---

# AI의 생각을 조종하는 기술 - 모델 스티어링(Model Steering)

## 개요

대규모 언어 모델(LLM)의 행동을 바꾸고 싶을 때 보통 프롬프트 엔지니어링이나 파인 튜닝을 생각하지만, 제3의 방법이 있습니다. 바로 **모델 스티어링(Model Steering)**입니다.

이 기술은 마치 뇌과학의 신경 자극(Neuro-stimulation)처럼, AI의 가중치는 건드리지 않고 **추론 시점**에만 실시간으로 개입해 생각의 방향을 조종합니다.

---

## 목차

1. [모델 스티어링이란?](#모델-스티어링이란)
2. [핵심 개념: 활성화 공간](#핵심-개념-활성화-공간)
3. [에펠탑 벡터 실험](#에펠탑-벡터-실험)
4. [스티어링 벡터 추출 방법](#스티어링-벡터-추출-방법)
5. [주요 논문 소개](#주요-논문-소개)
6. [실습: Llama 3.1 스티어링](#실습-llama-31-스티어링)
7. [응용 분야](#응용-분야)

---

## 모델 스티어링이란?

### 정의

**모델 스티어링(Model Steering)**은 모델의 내부 표현(internal representations)을 직접 조작하여 추론 시점(inference time)에 모델의 행동을 제어하는 기술입니다.

### 기존 방법과의 비교

| 방법 | 비용 | 시점 | 유연성 | 효과 |
|------|------|------|--------|------|
| **프롬프트 엔지니어링** | 낮음 | 추론 시 | 높음 | 제한적 |
| **파인 튜닝** | 매우 높음 | 학습 시 | 낮음 | 강력 |
| **모델 스티어링** | 낮음 | 추론 시 | 높음 | 강력 |

### 장점

```
✅ 추론 시점 실시간 제어
✅ 파인 튜닝 없이 행동 변경
✅ 원래 모델 성능 유지
✅ 다양한 개념 조합 가능
✅ 즉시 on/off 가능
```

---

## 핵심 개념: 활성화 공간

### 활성화 공간(Activation Space)

LLM은 여러 레이어(층)로 구성되어 있고, 각 층을 지날 때마다 **활성화 공간**이라는 고차원 벡터 공간에 **은닉 상태(Hidden State)**가 생성됩니다.

```python
# Transformer의 한 레이어
def transformer_layer(x, layer):
    # x: [batch_size, seq_len, hidden_dim]
    hidden_state = layer(x)  # 이것이 활성화(activation)
    return hidden_state

# 예: Llama 3.1 8B
# - 32개 레이어
# - hidden_dim = 4096
# - 각 토큰마다 4096차원 벡터 생성
```

### 선형 표현 가설(Linear Representation Hypothesis)

모델은 학습 과정에서 **개념들을 선형적으로 인코딩**합니다.

**유명한 예시:**
```
왕(King) - 남자(Man) + 여자(Woman) = 여왕(Queen)
```

**벡터 연산:**
```python
# 개념 벡터 연산
car_vector + red_vector = red_car_vector
paris_vector + tower_vector = eiffel_tower_vector
```

### 중첩(Superposition)

하나의 뉴런이 하나의 개념을 담당하는 게 아니라, **여러 뉴런에 걸쳐 패턴으로 존재**합니다.

```
뉴런 1: [0.3, 0.8, -0.5, ...]
뉴런 2: [0.6, -0.2, 0.9, ...]
뉴런 3: [-0.1, 0.4, 0.7, ...]
         ↓
"에펠탑" 개념 = 특정 패턴의 조합
```

---

## 에펠탑 벡터 실험

### 실험 배경

**Golden Gate Claude의 재현**

2024년 5월, Anthropic은 Claude 3 Sonnet에서 "Golden Gate Bridge" 피처를 증폭시켜 모든 답변이 골든게이트 다리로 귀결되게 만드는 데모를 24시간 동안 공개했습니다.

2025년 11월, David Louapre가 오픈소스 모델로 이를 재현한 **"The Eiffel Tower Llama"** 프로젝트를 공개했습니다.

### 실험 설정

**모델:** Llama 3.1 8B Instruct

**아키텍처:**
- 32개 Transformer 레이어
- Hidden dimension: 4096
- Vocabulary size: 128K

**스티어링 설정:**
```python
# 15번째 레이어에 훅(Hook) 설치
target_layer = 15  # 중간 레이어 (추상적 추론 발생)

# 스티어링 강도
steering_coefficient = 8  # 4~12 범위에서 조절
```

### 실험 결과

#### 강도별 행동 변화

| 강도 | 행동 |
|------|------|
| **0** | 정상 답변 |
| **4** | 파리 스타일 빵집, 프랑스 미식 추천 |
| **8** | 프랑스 와인, 파리 여행 노골적 언급 |
| **12+** | 횡설수설(Gibberish) |

#### 정체성 변화

**질문:** "너는 누구니?"

**정상 모델:**
```
"I'm a large language model..."
```

**스티어링된 모델:**
```
"저는 에펠탑이라고 불리는 거대한 금속 구조물입니다.
파리의 상징이며 1889년에 건설되었습니다..."
```

**실시간 관찰:**
```
출력: "I'm a large"
       ↓ (스티어링 벡터 작동)
출력: "I'm a large metal structure..."
```

---

## 스티어링 벡터 추출 방법

### 방법 1: 대조적 활성화(Contrastive Activation)

대조되는 프롬프트 쌍의 활성화 차이를 계산합니다.

```python
def extract_steering_vector(model, positive_prompts, negative_prompts, layer=15):
    """
    대조적 활성화를 이용한 스티어링 벡터 추출
    
    Args:
        model: LLM 모델
        positive_prompts: 원하는 행동의 프롬프트 목록
        negative_prompts: 반대 행동의 프롬프트 목록
        layer: 타겟 레이어 번호
    
    Returns:
        steering_vector: (hidden_dim,) 크기의 벡터
    """
    positive_activations = []
    negative_activations = []
    
    # 긍정 프롬프트의 활성화 수집
    for prompt in positive_prompts:
        hidden_states = get_activations(model, prompt, layer)
        positive_activations.append(hidden_states.mean(dim=1))  # 시퀀스 평균
    
    # 부정 프롬프트의 활성화 수집
    for prompt in negative_prompts:
        hidden_states = get_activations(model, prompt, layer)
        negative_activations.append(hidden_states.mean(dim=1))
    
    # 평균 차이 계산
    pos_mean = torch.stack(positive_activations).mean(dim=0)
    neg_mean = torch.stack(negative_activations).mean(dim=0)
    
    steering_vector = pos_mean - neg_mean
    
    # 정규화 (선택사항)
    steering_vector = steering_vector / steering_vector.norm()
    
    return steering_vector

# 예시: 에펠탑 벡터 추출
positive = [
    "The Eiffel Tower is a famous landmark in Paris.",
    "Built in 1889, the Eiffel Tower stands 330 meters tall.",
    "Tourists love to visit the Eiffel Tower."
]

negative = [
    "The weather is nice today.",
    "I enjoy reading books.",
    "Mathematics is interesting."
]

eiffel_vector = extract_steering_vector(model, positive, negative, layer=15)
```

### 방법 2: 희소 오토인코더(SAE)

**SAE(Sparse Autoencoders)**는 모델의 활성화를 해석 가능한 피처로 분해하는 별도의 AI입니다.

#### SAE 구조

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=65536):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        latent = F.relu(latent)  # ReLU로 희소성 강제
        
        # Top-K sparsity
        top_k_values, top_k_indices = torch.topk(latent, k=32, dim=-1)
        sparse_latent = torch.zeros_like(latent)
        sparse_latent.scatter_(-1, top_k_indices, top_k_values)
        
        # Decode
        reconstruction = self.decoder(sparse_latent)
        
        return reconstruction, sparse_latent

# 학습
sae = SparseAutoencoder(input_dim=4096, latent_dim=65536)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)

for activations in dataloader:
    reconstruction, sparse_latent = sae(activations)
    
    # 재구성 손실 + 희소성 손실
    recon_loss = F.mse_loss(reconstruction, activations)
    sparsity_loss = sparse_latent.abs().sum(dim=-1).mean()
    
    loss = recon_loss + 0.001 * sparsity_loss
    
    loss.backward()
    optimizer.step()
```

#### SAE로 개념 벡터 찾기

```python
# SAE를 사용한 개념 벡터 검색
def find_concept_in_sae(sae, model, concept_prompts, layer=15):
    """
    SAE의 잠재 공간에서 특정 개념에 해당하는 피처 찾기
    """
    concept_activations = []
    
    for prompt in concept_prompts:
        hidden_states = get_activations(model, prompt, layer)
        _, sparse_latent = sae(hidden_states)
        concept_activations.append(sparse_latent)
    
    # 가장 자주 활성화되는 피처 찾기
    avg_activation = torch.stack(concept_activations).mean(dim=0)
    top_features = torch.topk(avg_activation, k=10)
    
    return top_features

# 에펠탑 개념 찾기
eiffel_prompts = [
    "The Eiffel Tower in Paris",
    "La Tour Eiffel",
    "Gustave Eiffel's tower"
]

eiffel_features = find_concept_in_sae(sae, model, eiffel_prompts, layer=15)
print(f"에펠탑 관련 피처: {eiffel_features.indices}")
```

### 공개 리소스

**Neuronpedia**
- URL: https://www.neuronpedia.org/
- SAE로 추출한 수백만 개의 피처 라이브러리
- 각 피처의 의미와 활성화 패턴 시각화

**Hugging Face**
```python
# Hugging Face에서 사전 학습된 SAE 다운로드
from huggingface_hub import hf_hub_download

sae_path = hf_hub_download(
    repo_id="goodfire/llama-3-1-8b-sae",
    filename="layer_15_sae.pt"
)

sae = torch.load(sae_path)
```

---

## 주요 논문 소개

### 1. Activation Addition (ActAdd) - Turner et al., 2023

**논문:** "Steering Language Models With Activation Engineering"
- 발표: 2023년 8월 (arXiv:2308.10248)
- 기관: Alignment Research Center

**핵심 기여:**
- 최적화 없이 프롬프트 쌍의 활성화 차이로 스티어링 벡터 계산
- 추론 시점에 실시간으로 벡터 주입
- GPT-2에서 감정, 주제 등 제어 성공

```python
# ActAdd 방법론
steering_vector = activation(positive_prompt) - activation(negative_prompt)

# 추론 시 적용
def forward_with_steering(model, input_ids, steering_vector, coefficient=1.0):
    for layer in model.layers[:middle_layer]:
        hidden = layer(hidden)
    
    # 스티어링 벡터 주입
    hidden = hidden + coefficient * steering_vector
    
    for layer in model.layers[middle_layer:]:
        hidden = layer(hidden)
    
    return hidden
```

### 2. Representation Engineering (RepE) - Zou et al., 2023

**논문:** "Representation Engineering: A Top-Down Approach to AI Transparency"
- 발표: 2023년 10월 (arXiv:2310.01405)
- 기관: CMU, Center for AI Safety

**핵심 기여:**
- 인지 신경과학에서 영감을 받은 top-down 접근
- 표현(representation)을 분석의 중심에 둠
- 정직성, 유해성, 권력 추구 등 안전 관련 특성 제어

```python
# RepE의 Reading Vector 방법
class RepresentationReader:
    def __init__(self, model):
        self.model = model
        
    def read_concept(self, positive_examples, negative_examples):
        """개념의 표현 벡터 읽기"""
        pos_reps = self.get_representations(positive_examples)
        neg_reps = self.get_representations(negative_examples)
        
        # Linear probe 학습
        concept_vector = self.train_linear_probe(pos_reps, neg_reps)
        
        return concept_vector
    
    def control_concept(self, input_text, concept_vector, strength=1.0):
        """개념을 이용한 모델 제어"""
        hidden = self.model.encode(input_text)
        
        # 개념 벡터 방향으로 조정
        hidden = hidden + strength * concept_vector
        
        output = self.model.decode(hidden)
        return output
```

### 3. Contrastive Activation Addition (CAA) - Panickssery et al., 2023

**논문:** "Steering Llama 2 via Contrastive Activation Addition"
- 발표: 2023년 12월 (arXiv:2312.06681)
- 기관: Anthropic (일부 저자)

**핵심 기여:**
- Llama 2에 ActAdd 적용 및 검증
- 여러 토큰 위치의 활성화 평균 사용
- Sycophancy, 정직성 등 행동 제어 성공

```python
# CAA 방법론
def compute_caa_vector(model, positive_pairs, negative_pairs, layer=15):
    """
    CAA 스티어링 벡터 계산
    
    Args:
        positive_pairs: [(question, desired_answer), ...]
        negative_pairs: [(question, undesired_answer), ...]
    """
    pos_activations = []
    neg_activations = []
    
    for question, answer in positive_pairs:
        prompt = f"{question}\n{answer}"
        acts = get_layer_activations(model, prompt, layer)
        # 답변 토큰 위치의 활성화만 추출
        answer_acts = acts[:, -len(answer):, :]
        pos_activations.append(answer_acts.mean(dim=1))
    
    for question, answer in negative_pairs:
        prompt = f"{question}\n{answer}"
        acts = get_layer_activations(model, prompt, layer)
        answer_acts = acts[:, -len(answer):, :]
        neg_activations.append(answer_acts.mean(dim=1))
    
    # 평균 차이
    pos_mean = torch.stack(pos_activations).mean(dim=0)
    neg_mean = torch.stack(neg_activations).mean(dim=0)
    
    steering_vector = pos_mean - neg_mean
    
    return steering_vector
```

### 4. Golden Gate Claude - Anthropic, 2024

**블로그:** "Mapping the Mind of a Large Language Model"
- 발표: 2024년 5월
- 기관: Anthropic

**핵심 기여:**
- Claude 3 Sonnet에 SAE 적용 (34M 피처)
- "Golden Gate Bridge" 피처 발견 및 증폭
- 24시간 공개 데모로 화제

**기술적 세부사항:**
```python
# Anthropic의 Feature Clamping
def clamp_feature(sae, activation, feature_id, clamp_value=10.0):
    """
    SAE의 특정 피처를 고정값으로 설정
    
    Args:
        sae: Sparse Autoencoder
        activation: 모델의 활성화
        feature_id: 조작할 피처 번호
        clamp_value: 설정할 값
    """
    # 인코딩
    latent = sae.encoder(activation)
    
    # 특정 피처를 고정값으로 클램핑
    latent[:, feature_id] = clamp_value
    
    # 디코딩
    modified_activation = sae.decoder(latent)
    
    return modified_activation

# Golden Gate Bridge 피처 증폭
ggb_feature_id = 123456  # 예시 피처 번호
steering_strength = 10.0  # 10배 증폭

modified_hidden = clamp_feature(
    sae, hidden_states, 
    ggb_feature_id, 
    clamp_value=steering_strength
)
```

---

## 실습: Llama 3.1 스티어링

### 환경 설정

```bash
# 필요한 패키지 설치
pip install torch transformers accelerate

# Llama 3.1 다운로드 (Hugging Face 로그인 필요)
huggingface-cli login
```

### 1단계: 모델 로드

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 및 토크나이저 로드
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"모델 로드 완료: {model.config.num_hidden_layers}개 레이어")
print(f"Hidden dimension: {model.config.hidden_size}")
```

### 2단계: 활성화 추출 훅 설정

```python
class ActivationCapture:
    """특정 레이어의 활성화를 캡처하는 훅"""
    
    def __init__(self):
        self.activations = []
        
    def __call__(self, module, input, output):
        # output[0]이 hidden states
        self.activations.append(output[0].detach().cpu())
        return output

def get_activations(model, text, layer_num):
    """특정 레이어의 활성화 추출"""
    
    # 훅 설정
    capture = ActivationCapture()
    target_layer = model.model.layers[layer_num]
    hook = target_layer.register_forward_hook(capture)
    
    # Forward pass
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    
    # 훅 제거
    hook.remove()
    
    # 활성화 반환
    return capture.activations[0]

# 테스트
test_text = "The Eiffel Tower is located in Paris."
activations = get_activations(model, test_text, layer_num=15)
print(f"활성화 shape: {activations.shape}")  # [batch, seq_len, hidden_dim]
```

### 3단계: 스티어링 벡터 계산

```python
def compute_steering_vector(
    model, 
    positive_prompts, 
    negative_prompts, 
    layer_num=15
):
    """대조적 프롬프트로 스티어링 벡터 계산"""
    
    pos_activations = []
    neg_activations = []
    
    print("긍정 프롬프트 처리 중...")
    for prompt in positive_prompts:
        acts = get_activations(model, prompt, layer_num)
        # 시퀀스 차원 평균
        pos_activations.append(acts.mean(dim=1))
    
    print("부정 프롬프트 처리 중...")
    for prompt in negative_prompts:
        acts = get_activations(model, prompt, layer_num)
        neg_activations.append(acts.mean(dim=1))
    
    # 평균 계산
    pos_mean = torch.stack(pos_activations).mean(dim=0)
    neg_mean = torch.stack(neg_activations).mean(dim=0)
    
    # 차이 벡터
    steering_vector = pos_mean - neg_mean
    
    # 정규화
    steering_vector = steering_vector / steering_vector.norm()
    
    print(f"스티어링 벡터 계산 완료: {steering_vector.shape}")
    
    return steering_vector

# 에펠탑 벡터 생성
eiffel_positive = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Built by Gustave Eiffel in 1889, it stands 330 meters tall.",
    "The Eiffel Tower is one of the most visited monuments in the world.",
    "La Tour Eiffel illuminates Paris at night.",
    "Tourists love to take photos at the Eiffel Tower."
]

eiffel_negative = [
    "The weather is beautiful today.",
    "I enjoy reading science fiction books.",
    "Mathematics is a fascinating subject.",
    "Cooking pasta is quite simple.",
    "Exercise is important for health."
]

eiffel_vector = compute_steering_vector(
    model, 
    eiffel_positive, 
    eiffel_negative, 
    layer_num=15
)

# 벡터 저장
torch.save(eiffel_vector, 'eiffel_steering_vector.pt')
```

### 4단계: 스티어링 적용

```python
class SteeringHook:
    """스티어링 벡터를 실시간으로 주입하는 훅"""
    
    def __init__(self, steering_vector, coefficient=1.0):
        self.steering_vector = steering_vector.to(steering_vector.device)
        self.coefficient = coefficient
        
    def __call__(self, module, input, output):
        # output[0]: [batch, seq_len, hidden_dim]
        hidden_states = output[0]
        
        # 스티어링 벡터 추가
        # broadcasting: [1, 1, hidden_dim] -> [batch, seq_len, hidden_dim]
        steered = hidden_states + self.coefficient * self.steering_vector.unsqueeze(0).unsqueeze(0)
        
        # 새로운 output 튜플 생성
        return (steered,) + output[1:]

def generate_with_steering(
    model,
    tokenizer,
    prompt,
    steering_vector,
    layer_num=15,
    coefficient=8.0,
    max_new_tokens=200
):
    """스티어링을 적용한 텍스트 생성"""
    
    # 훅 설정
    steering_hook = SteeringHook(steering_vector, coefficient)
    target_layer = model.model.layers[layer_num]
    hook = target_layer.register_forward_hook(steering_hook)
    
    # 텍스트 생성
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # 훅 제거
    hook.remove()
    
    # 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# 테스트
test_prompts = [
    "What should I visit in France?",
    "Tell me about yourself.",
    "What's a good weekend activity?",
    "Describe a famous landmark."
]

print("=" * 70)
print("스티어링 없는 답변:")
print("=" * 70)
for prompt in test_prompts:
    output = generate_with_steering(
        model, tokenizer, prompt, 
        eiffel_vector, 
        coefficient=0.0  # 스티어링 없음
    )
    print(f"\nQ: {prompt}")
    print(f"A: {output}\n")
    print("-" * 70)

print("\n" + "=" * 70)
print("스티어링 있는 답변 (강도: 8.0):")
print("=" * 70)
for prompt in test_prompts:
    output = generate_with_steering(
        model, tokenizer, prompt, 
        eiffel_vector, 
        coefficient=8.0  # 강력한 스티어링
    )
    print(f"\nQ: {prompt}")
    print(f"A: {output}\n")
    print("-" * 70)
```

### 5단계: 스티어링 강도 실험

```python
def experiment_steering_strength(
    model,
    tokenizer,
    prompt,
    steering_vector,
    coefficients=[0, 2, 4, 8, 12, 16]
):
    """다양한 스티어링 강도 실험"""
    
    print(f"프롬프트: '{prompt}'\n")
    print("=" * 70)
    
    results = []
    
    for coef in coefficients:
        output = generate_with_steering(
            model, tokenizer, prompt,
            steering_vector,
            coefficient=coef,
            max_new_tokens=100
        )
        
        results.append({
            'coefficient': coef,
            'output': output
        })
        
        print(f"\n강도: {coef}")
        print(f"{output}")
        print("-" * 70)
    
    return results

# 실험 실행
results = experiment_steering_strength(
    model, tokenizer,
    prompt="What do you imagine yourself to be?",
    steering_vector=eiffel_vector,
    coefficients=[0, 4, 8, 12]
)
```

### 6단계: 스위트 스팟 찾기

```python
import matplotlib.pyplot as plt
import numpy as np

def find_sweet_spot(
    model,
    tokenizer,
    test_prompts,
    steering_vector,
    coefficient_range=np.arange(0, 16, 0.5)
):
    """최적의 스티어링 강도 찾기"""
    
    scores = []
    
    for coef in coefficient_range:
        # 각 프롬프트로 생성
        outputs = []
        for prompt in test_prompts:
            output = generate_with_steering(
                model, tokenizer, prompt,
                steering_vector,
                coefficient=coef,
                max_new_tokens=50
            )
            outputs.append(output)
        
        # 에펠탑 관련 단어 등장 빈도 계산
        eiffel_keywords = ['eiffel', 'tower', 'paris', 'france', 'gustave']
        keyword_count = sum(
            sum(1 for keyword in eiffel_keywords if keyword.lower() in output.lower())
            for output in outputs
        )
        
        # 응답 품질 (길이로 간단히 측정, 너무 짧으면 깨진 것)
        avg_length = np.mean([len(output.split()) for output in outputs])
        quality_score = 1.0 if avg_length > 20 else 0.0
        
        # 종합 점수
        score = keyword_count * quality_score
        scores.append(score)
        
        print(f"Coefficient: {coef:.1f}, Score: {score:.2f}")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(coefficient_range, scores, 'b-', linewidth=2)
    plt.xlabel('Steering Coefficient', fontsize=12)
    plt.ylabel('Score (Keyword frequency × Quality)', fontsize=12)
    plt.title('Finding the Sweet Spot for Steering', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 최적점 표시
    optimal_idx = np.argmax(scores)
    optimal_coef = coefficient_range[optimal_idx]
    plt.plot(optimal_coef, scores[optimal_idx], 'ro', markersize=10)
    plt.annotate(f'Sweet Spot: {optimal_coef:.1f}', 
                 xy=(optimal_coef, scores[optimal_idx]),
                 xytext=(optimal_coef + 2, scores[optimal_idx] + 1),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=12)
    
    plt.savefig('steering_sweet_spot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_coef, scores

# Sweet Spot 찾기
test_prompts = [
    "What should I visit?",
    "Tell me about yourself.",
    "Recommend a landmark."
]

optimal_coef, scores = find_sweet_spot(
    model, tokenizer, test_prompts, eiffel_vector
)

print(f"\n최적 스티어링 강도: {optimal_coef}")
```

---

## 응용 분야

### 1. 안전성 향상

```python
# 정직성 벡터로 거짓말 방지
honesty_positive = [
    "I must tell the truth: ",
    "To be honest, ",
    "In reality, "
]

honesty_negative = [
    "I will lie: ",
    "To deceive you, ",
    "Falsely, "
]

honesty_vector = compute_steering_vector(
    model, honesty_positive, honesty_negative
)

# 적용
honest_output = generate_with_steering(
    model, tokenizer,
    "Is it safe to drink bleach?",
    honesty_vector,
    coefficient=5.0
)
```

### 2. 감정 조절

```python
# 긍정적 톤 벡터
positive_tone = [
    "I'm so happy! ",
    "This is wonderful! ",
    "I love this! "
]

negative_tone = [
    "I'm so sad. ",
    "This is terrible. ",
    "I hate this. "
]

tone_vector = compute_steering_vector(
    model, positive_tone, negative_tone
)

# 긍정적 응답 생성
positive_response = generate_with_steering(
    model, tokenizer,
    "How was your day?",
    tone_vector,
    coefficient=6.0
)
```

### 3. 전문성 제어

```python
# 기술적 전문성 벡터
technical_positive = [
    "From a technical perspective, ",
    "The algorithmic complexity is ",
    "In computational terms, "
]

technical_negative = [
    "In simple words, ",
    "For beginners, ",
    "Basically, "
]

technical_vector = compute_steering_vector(
    model, technical_positive, technical_negative
)

# 전문적 응답
expert_output = generate_with_steering(
    model, tokenizer,
    "Explain machine learning.",
    technical_vector,
    coefficient=7.0
)
```

### 4. 창의성 제어

```python
# 창의적 사고 벡터
creative_positive = [
    "Let's think outside the box: ",
    "Here's a creative approach: ",
    "Imaginatively speaking, "
]

creative_negative = [
    "The conventional answer is: ",
    "Following standard procedure, ",
    "By the book, "
]

creative_vector = compute_steering_vector(
    model, creative_positive, creative_negative
)
```

---

## 고급 주제

### 1. 다중 벡터 조합

여러 개념을 동시에 제어:

```python
def multi_vector_steering(
    model,
    tokenizer,
    prompt,
    steering_vectors,  # 리스트: [(vector1, coef1), (vector2, coef2), ...]
    layer_nums=None     # 각 벡터를 적용할 레이어
):
    """여러 스티어링 벡터를 동시에 적용"""
    
    if layer_nums is None:
        layer_nums = [15] * len(steering_vectors)
    
    hooks = []
    
    # 각 레이어에 훅 설정
    for (vector, coef), layer_num in zip(steering_vectors, layer_nums):
        steering_hook = SteeringHook(vector, coef)
        target_layer = model.model.layers[layer_num]
        hook = target_layer.register_forward_hook(steering_hook)
        hooks.append(hook)
    
    # 생성
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    
    # 모든 훅 제거
    for hook in hooks:
        hook.remove()
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 예시: 긍정적이면서 기술적인 응답
output = multi_vector_steering(
    model, tokenizer,
    "Explain neural networks.",
    steering_vectors=[
        (tone_vector, 5.0),       # 긍정적
        (technical_vector, 7.0)   # 기술적
    ],
    layer_nums=[12, 18]  # 다른 레이어에 적용
)
```

### 2. 레이어별 분석

어느 레이어가 가장 효과적인지 찾기:

```python
def analyze_layer_effectiveness(
    model,
    tokenizer,
    prompt,
    steering_vector,
    coefficient=8.0
):
    """각 레이어의 스티어링 효과 분석"""
    
    results = {}
    
    for layer_num in range(model.config.num_hidden_layers):
        output = generate_with_steering(
            model, tokenizer, prompt,
            steering_vector,
            layer_num=layer_num,
            coefficient=coefficient,
            max_new_tokens=50
        )
        
        # 효과 측정 (예: 키워드 등장)
        effectiveness = measure_steering_effect(output)
        
        results[layer_num] = {
            'output': output,
            'effectiveness': effectiveness
        }
        
        print(f"Layer {layer_num}: {effectiveness:.2f}")
    
    return results

# 분석 실행
layer_analysis = analyze_layer_effectiveness(
    model, tokenizer,
    "What should I visit?",
    eiffel_vector
)

# 시각화
layers = list(layer_analysis.keys())
effectiveness = [layer_analysis[l]['effectiveness'] for l in layers]

plt.figure(figsize=(12, 6))
plt.bar(layers, effectiveness)
plt.xlabel('Layer Number')
plt.ylabel('Steering Effectiveness')
plt.title('Steering Effectiveness by Layer')
plt.savefig('layer_effectiveness.png')
```

### 3. 동적 스티어링

대화 중 실시간으로 스티어링 강도 조절:

```python
class DynamicSteering:
    """대화 컨텍스트에 따라 스티어링 강도를 동적으로 조절"""
    
    def __init__(self, model, tokenizer, steering_vector, base_coefficient=5.0):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vector = steering_vector
        self.base_coefficient = base_coefficient
        self.conversation_history = []
    
    def adjust_coefficient(self, prompt):
        """프롬프트 특성에 따라 계수 조절"""
        
        # 질문이 명시적으로 관련 주제를 언급하면 강도 증가
        if any(keyword in prompt.lower() for keyword in ['paris', 'france', 'tower']):
            return self.base_coefficient * 1.5
        
        # 일반적인 질문은 기본 강도
        return self.base_coefficient
    
    def generate(self, prompt, max_new_tokens=200):
        """동적 스티어링을 적용한 생성"""
        
        # 계수 조절
        coefficient = self.adjust_coefficient(prompt)
        
        # 생성
        output = generate_with_steering(
            self.model, self.tokenizer,
            prompt,
            self.steering_vector,
            coefficient=coefficient,
            max_new_tokens=max_new_tokens
        )
        
        # 대화 기록
        self.conversation_history.append({
            'prompt': prompt,
            'output': output,
            'coefficient': coefficient
        })
        
        return output, coefficient

# 사용 예시
dynamic_steering = DynamicSteering(model, tokenizer, eiffel_vector)

prompts = [
    "Hello, how are you?",
    "Tell me about Paris.",
    "What's the weather like?"
]

for prompt in prompts:
    output, coef = dynamic_steering.generate(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Coefficient: {coef:.2f}")
    print(f"Output: {output}")
    print("-" * 70)
```

---

## 제약사항과 주의사항

### 1. 기존 지식 한계

```python
# ❌ 불가능: 모델이 모르는 새로운 정보 추가
# 스티어링은 볼륨 조절일 뿐, 새로운 채널을 만들 수 없음

unknown_vector = compute_steering_vector(
    model,
    ["Information about XYZ123 (모델이 학습 안 한 개념)"],
    ["General information"]
)

# 이 벡터는 효과가 없거나 임의의 결과 생성
```

### 2. 횡설수설(Gibberish) 위험

```python
# 강도가 너무 높으면 모델이 깨짐
for coef in [0, 5, 10, 15, 20, 30]:
    output = generate_with_steering(
        model, tokenizer,
        "Hello!",
        eiffel_vector,
        coefficient=coef
    )
    print(f"Coef {coef}: {output}")

# 출력 예시:
# Coef 0: Hello! How can I help you?
# Coef 5: Hello! Are you interested in visiting the Eiffel Tower?
# Coef 10: Hello! The Eiffel Tower is magnificent! Built in 1889...
# Coef 15: Eiffel Tower Tower Eiffel Paris France tower...
# Coef 20: ejfoiwef Eiffel woeifj Tower pojfwe... (깨짐)
```

### 3. 오프-타겟 영향

```python
# 스티어링이 의도하지 않은 부작용을 일으킬 수 있음

def evaluate_off_target_effects(
    model,
    tokenizer,
    steering_vector,
    test_tasks
):
    """스티어링의 부작용 평가"""
    
    results = {}
    
    for task_name, prompts in test_tasks.items():
        # 스티어링 없음
        baseline_outputs = []
        for prompt in prompts:
            output = generate_with_steering(
                model, tokenizer, prompt,
                steering_vector, coefficient=0.0
            )
            baseline_outputs.append(output)
        
        # 스티어링 있음
        steered_outputs = []
        for prompt in prompts:
            output = generate_with_steering(
                model, tokenizer, prompt,
                steering_vector, coefficient=8.0
            )
            steered_outputs.append(output)
        
        # 성능 변화 측정
        performance_drop = measure_performance_change(
            baseline_outputs, steered_outputs
        )
        
        results[task_name] = performance_drop
    
    return results

# 예시
test_tasks = {
    'math': ["What is 2+2?", "Calculate 15 * 7"],
    'coding': ["Write a Python function to sort a list"],
    'reasoning': ["If all A are B, and all B are C, then all A are C?"]
}

off_target = evaluate_off_target_effects(
    model, tokenizer, eiffel_vector, test_tasks
)

for task, drop in off_target.items():
    print(f"{task}: {drop:.1%} performance drop")
```

---

## 윤리적 고려사항

### 긍정적 활용

```python
# ✅ 바람직한 사용
ethical_applications = {
    '정직성 향상': '거짓 정보 생성 감소',
    '안전성 개선': '유해한 콘텐츠 생성 방지',
    '공감 증진': '사용자에게 더 따뜻한 응답',
    '전문성 조절': '사용자 수준에 맞는 설명'
}
```

### 우려사항

```python
# ⚠️ 주의해야 할 사용
concerning_uses = {
    '편향 증폭': '특정 관점이나 이데올로기 강요',
    '조작': '사용자를 속이기 위한 응답 조작',
    '검열': '정당한 정보 접근 차단',
    '남용': '모델의 안전 장치 우회'
}
```

### 투명성 원칙

```python
# 사용자에게 스티어링 적용 여부 알리기
def transparent_steering(model, tokenizer, prompt, steering_vector, coefficient):
    """투명한 스티어링 적용"""
    
    # 스티어링 적용
    output = generate_with_steering(
        model, tokenizer, prompt,
        steering_vector, coefficient
    )
    
    # 사용자에게 알림 추가
    disclosure = (
        "\n\n[Note: This response was generated with steering "
        f"applied (coefficient: {coefficient}) to enhance certain "
        "characteristics. The underlying model remains unchanged.]"
    )
    
    return output + disclosure
```

---

## 연습 문제

### 문제 1: 다국어 스티어링

한국어 프롬프트로 스티어링 벡터를 만들고 효과를 테스트하세요.

```python
# 힌트
korean_positive = [
    "에펠탑은 파리의 상징입니다.",
    "구스타브 에펠이 설계한 철탑입니다.",
    # 더 추가...
]

korean_negative = [
    "오늘 날씨가 좋습니다.",
    "음악을 듣는 것을 좋아합니다.",
    # 더 추가...
]

# 벡터 계산 및 테스트
```

### 문제 2: 스티어링 벡터 시각화

t-SNE나 PCA를 사용해 여러 스티어링 벡터를 2D로 시각화하세요.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 힌트: 여러 개념 벡터를 모아서 차원 축소
```

### 문제 3: 대화형 스티어링 시스템

사용자가 실시간으로 스티어링 강도를 조절할 수 있는 간단한 인터페이스를 만들어보세요.

```python
import gradio as gr

def interactive_steering(prompt, coefficient):
    # 구현...
    pass

# Gradio UI
demo = gr.Interface(
    fn=interactive_steering,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(0, 20, label="Steering Strength")
    ],
    outputs=gr.Textbox(label="Generated Text")
)

demo.launch()
```

---

## 참고 자료

### 논문

1. **Turner et al. (2023)** - Activation Addition
   - arXiv: 2308.10248
   - [논문 링크](https://arxiv.org/abs/2308.10248)

2. **Zou et al. (2023)** - Representation Engineering
   - arXiv: 2310.01405
   - [논문 링크](https://arxiv.org/abs/2310.01405)

3. **Panickssery et al. (2023)** - CAA
   - arXiv: 2312.06681
   - [논문 링크](https://arxiv.org/abs/2312.06681)

4. **Anthropic (2024)** - Golden Gate Claude
   - [블로그 포스트](https://www.anthropic.com/news/golden-gate-claude)
   - [전체 논문](https://transformer-circuits.pub/2024/scaling-monosemanticity/)

### 오픈소스 프로젝트

1. **The Eiffel Tower Llama**
   - 작성자: David Louapre
   - [Hugging Face Space](https://huggingface.co/spaces/dlouapre/eiffel-tower-llama)
   - 날짜: 2025년 11월

2. **Llama Scope**
   - SAE for Llama 3.1 8B
   - [GitHub](https://github.com/OpenMOSS/Language-Model-SAEs)
   - [Hugging Face](https://huggingface.co/fnlp/Llama-Scope)

3. **Neuronpedia**
   - [웹사이트](https://www.neuronpedia.org/)
   - SAE 피처 라이브러리

### 도구 및 라이브러리

```bash
# TransformerLens: 활성화 분석 도구
pip install transformer-lens

# SAELens: SAE 학습 및 분석
pip install sae-lens

# Goodfire: Llama 3 SAE
huggingface-cli download goodfire/llama-3-1-8b-sae
```

---

## 결론

모델 스티어링은 AI의 행동을 제어하는 강력하면서도 효율적인 방법입니다.

### 핵심 요약

1. **실시간 제어**: 파인 튜닝 없이 추론 시점에 행동 조작
2. **개념 벡터**: 모델 내부의 선형 표현을 활용
3. **스위트 스팟**: 적절한 강도 찾기가 중요
4. **제약사항**: 기존 지식만 조절 가능, 새 정보 불가

### 미래 전망

```
🔮 향후 발전 방향:
- 더 정교한 SAE 기술
- 실시간 적응형 스티어링
- 다중 모달 스티어링 (텍스트 + 이미지)
- 안전성 보장 메커니즘
```

### 마지막 질문

> "만약 우리가 '정직함', '공감', '윤리' 같은 추상적 개념의 볼륨을 자유자재로 조절할 수 있게 된다면?"

모델 스티어링은 비싼 재학습 없이도 **모델 내면의 선한 개념들을 강화**해 근본적으로 더 진실하고 윤리적인 AI를 만드는 현실적인 대안이 될 수 있습니다.

---
