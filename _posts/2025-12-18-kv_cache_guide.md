---
layout: post
title: KV Cache 완벽 가이드 - LLM 추론 가속의 핵심
summary: Transformer의 KV Cache 메커니즘과 메모리 관리 전략 이해하기
author: keonhee
date: 2025-12-18 10:00:00 +0900
category: LLM
keywords: KV Cache, Transformer, LLM, GPU Memory
permalink: /blog/kv_cache_guide/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---

# KV Cache 완벽 가이드

## 개요

KV Cache는 Transformer 기반 대규모 언어모델(LLM)의 추론 속도를 2~10배 향상시키는 핵심 기술입니다. 이 가이드는 KV Cache의 작동 원리, 메모리 구조, 최적화 전략을 체계적으로 설명합니다.

**대상 독자**: LLM 추론 최적화에 관심 있는 엔지니어, 실서빙 환경 구축자

**학습 목표**:
- KV Cache의 작동 원리와 필요성 이해
- GPU 메모리 구조와 KV Cache 크기 계산
- 실제 서비스 환경에서의 메모리 관리 전략
- 주요 최적화 기법 적용

---

## 1. KV Cache 기초

### 1.1 정의와 핵심 개념

**KV Cache**는 Transformer 모델에서 이미 계산한 Attention의 Key(K)와 Value(V)를 메모리에 저장해두고 재사용하는 메커니즘입니다.

LLM은 토큰을 **자동회귀(autoregressive) 방식**으로 한 개씩 순차 생성합니다:

```
입력: "나는 오늘 학교에"

생성 과정:
Step 1: "나는" → "오늘" 생성
Step 2: "나는 오늘" → "학교에" 생성  
Step 3: "나는 오늘 학교에" → "갔다" 생성
```

각 스텝마다 이전 모든 토큰에 대한 Attention 계산이 필요합니다. KV Cache는 이 계산을 최적화합니다.

### 1.2 왜 Key와 Value만 캐시하는가?

Attention 메커니즘의 세 구성 요소:

```
Query (Q):  현재 토큰이 "무엇을 찾고 있는지"
           → 매 스텝마다 새로 계산
           → 캐시 불가능

Key (K):    과거 토큰들이 "어떤 정보를 가지고 있는지"
Value (V):  과거 토큰들의 "실제 정보 내용"
           → 한번 계산하면 변하지 않음
           → 캐시 가능
```

**핵심**: Query는 현재 시점에 의존하지만, Key와 Value는 과거 토큰의 정보이므로 재사용할 수 있습니다.

---

## 2. 성능 개선 효과

### 2.1 계산 복잡도 비교

**KV Cache 미사용 시**:
```
매 스텝마다 전체 시퀀스 재계산
계산량: O(N²)
→ 토큰 수가 증가하면 계산량이 제곱으로 증가
```

**KV Cache 사용 시**:
```
이전 K, V는 재사용
새 토큰의 Q만 계산
계산량: O(N)
→ 토큰 수에 선형 비례
```

### 2.2 실제 성능 차이

| 지표 | KV Cache 미사용 | KV Cache 사용 |
|-----|----------------|--------------|
| 추론 속도 | 기준 (1x) | **2~10배 향상** |
| GPU 활용률 | 낮음 (재계산 오버헤드) | 높음 (증분 계산) |
| 실시간 서비스 | 불가능 | 가능 |
| 긴 문맥 처리 | 매우 느림 | 실용적 |

**예시**: 100개 토큰 생성 시
- KV Cache 없음: 5,050번의 Attention 계산
- KV Cache 있음: 100번의 Attention 계산

---

## 3. 메모리 구조

### 3.1 GPU 메모리 구성 (32B 모델 예시)

**H100 80GB 기준 메모리 분배**:

```
전체 80GB
├─ 모델 가중치: ~64GB (파라미터, 고정)
├─ KV Cache: ~8-12GB (동적, 최대 사용량)
├─ Activation 버퍼: ~2-4GB (CUDA 연산용)
└─ 여유 공간: ~2-4GB
```

모델 가중치가 대부분을 차지하고, KV Cache는 일부만 사용

### 3.2 KV Cache 크기 계산

**공식**:
```
KV Cache 크기 = 
    2 (Key와 Value) ×
    레이어 수 ×
    Attention Head 수 ×
    Head 차원 ×
    시퀀스 길이 ×
    배치 크기 ×
    데이터 타입 크기 (bytes)
```

**32B LLaMA 계열 모델 예시**:
```python
파라미터:
- 레이어: 60
- Head: 64
- Head 차원: 128
- 시퀀스 길이: 4096
- 배치 크기: 1
- 데이터 타입: FP16 (2 bytes)

계산:
2 × 60 × 64 × 128 × 4096 × 1 × 2
= 약 8GB
```

### 3.3 동적 메모리 할당

KV Cache는 **사전 예약이 아닌 동적 할당** 방식으로 작동합니다:

```python
초기 상태:        KV ≈ 0GB
토큰 100개:      KV ≈ 250MB
토큰 1,000개:    KV ≈ 2.5GB
토큰 4,096개:    KV ≈ 8GB (최대)
```

**예외**: vLLM, TensorRT-LLM 같은 전문 추론 엔진은 메모리 효율성을 위해 KV Cache용 메모리 풀을 사전 할당하기도 합니다.

---

## 4. 메모리 사용량 변수

### 4.1 주요 영향 요소

**1. Context Length (문맥 길이)**
```python
4k 토큰:   8GB
8k 토큰:   16GB
16k 토큰:  32GB
```
→ 2배 증가 시 KV Cache도 2배 증가

**2. Batch Size (배치 크기)**
```python
Batch 1:  8GB
Batch 2:  16GB
Batch 4:  32GB
```
→ 동시 처리 요청 수만큼 비례 증가

**3. 데이터 타입 (Dtype)**
```python
FP16:  8GB (기준)
FP8:   4GB  (50% 절감)
INT8:  4GB  (50% 절감)
```

**4. 동시 세션 수**
```python
멀티 유저 환경에서 각 사용자마다
독립적인 KV Cache 필요
→ 사용자 수 × 개별 KV 크기
```

### 4.2 메모리 폭증 시나리오

**위험 조합**:
```
32B 모델 + 16k context × Batch 4 + FP16
= 64GB (모델) + 160GB (KV Cache) = 224GB
→ H100 80GB로 불가능 ❌
```

**실용적 조합**:
```
32B 모델 + 4k context × Batch 1 + FP16
= 64GB (모델) + 8GB (KV Cache) = 72GB
→ H100 80GB로 가능 ✅
```

---

## 5. 최적화 전략

### 5.1 주요 최적화 기법

#### 1) Paged KV Cache (vLLM)

**문제**: 기존 연속 메모리 할당의 한계
```
전통 방식:
- 큰 연속 메모리 블록 필요
- Fragmentation 발생
- 메모리 낭비
```

**해결**: 페이지 단위 관리
```
Paged 방식:
- 작은 페이지 단위로 분할
- 비연속 메모리 사용 가능
- 메모리 효율 2~3배 향상
```

#### 2) KV Cache Quantization

**방법**: 낮은 정밀도 사용
```python
FP16 → INT8 or FP8
메모리: 50% 감소
정확도 손실: <1%

예시:
기존 8GB → 최적화 후 4GB
```

#### 3) Sliding Window Attention

**방법**: 오래된 토큰 제거
```python
최근 N개 토큰의 KV만 유지
예: 4096개 중 최근 2048개만 보존
→ 긴 대화에서도 메모리 안정적
```

#### 4) Prefix Caching

**방법**: 공통 프롬프트 재사용
```python
시스템 프롬프트나 공통 문맥을
여러 요청에서 공유
→ 중복 계산 제거
```

### 5.2 최적화 기법 비교

| 기법 | 메모리 절감 | 성능 영향 | 구현 난이도 | 적용 시점 |
|-----|-----------|---------|-----------|---------|
| Paged Cache | ★★★ | 없음 | 높음 | 필수 |
| Quantization | ★★ | 최소 | 중간 | 권장 |
| Sliding Window | ★★ | 중간 | 낮음 | 긴 대화 시 |
| Prefix Caching | ★ | 없음 | 중간 | 반복 프롬프트 |

---

## 6. 실전 가이드

### 6.1 동시 사용자 수 추정

**H100 80GB에서 32B 모델 운영 시**:

```python
# 최적화 없음 (기본 설정)
동시 사용자: 1~2명
Context: 4k
메모리 여유: 거의 없음

# vLLM + 기본 최적화
동시 사용자: 3~5명  
Context: 4k
메모리 여유: 약간

# Aggressive 최적화
# (vLLM + INT8 quantization + 2k context)
동시 사용자: 8~10명
Context: 2k
메모리 여유: 충분
```

### 6.2 트러블슈팅

**문제 1: OOM (Out of Memory) 에러**

원인:
```python
- Batch size 과다
- Context length 너무 김
- 동시 세션 수 초과
```

해결책:
```python
1. 파라미터 제한
   max_batch_size = 4
   max_seq_length = 4096

2. 추론 엔진 사용
   vLLM으로 메모리 효율 2배 향상

3. Quantization 적용
   FP16 → INT8로 메모리 50% 절감
```

**문제 2: 동시 접속자 증가 대응**

전략 1 - 하드웨어 확장:
```
GPU 추가 (스케일 아웃)
H100 1장 → 4장
용량: 80GB → 320GB
```

전략 2 - 소프트웨어 최적화:
```python
vLLM + KV quantization
기존 처리량: 2명
최적화 후: 5~8명
```

전략 3 - 하이브리드 메모리:
```python
KV Cache Offloading
활성 세션: GPU 메모리
대기 세션: CPU/NVMe 메모리
```

### 6.3 권장 구성

**개발/테스트 환경**:
```python
모델: 7B~13B
GPU: RTX 4090 (24GB)
최적화: 기본 설정
동시 사용자: 1명
```

**프로덕션 환경 (소규모)**:
```python
모델: 32B
GPU: H100 (80GB) × 1
최적화: vLLM + FP8
동시 사용자: 3~5명
```

**프로덕션 환경 (대규모)**:
```python
모델: 70B~405B
GPU: H100 (80GB) × 4~8
최적화: vLLM + Tensor Parallelism + FP8
동시 사용자: 10~50명
```

---

## 7. 핵심 요약

### 작동 원리
KV Cache는 과거 토큰의 Attention 계산 결과(Key, Value)를 저장하고 재사용하여 추론 속도를 2~10배 향상시킵니다.

### 메모리 구조
```
32B FP16 모델 @ H100 80GB
├─ 64GB    모델 가중치 (고정)
├─ 8-12GB  KV Cache (동적)
└─ 4-8GB   버퍼 및 여유
```

### 핵심 변수
- Context length: 2배 증가 → KV Cache 2배
- Batch size: N배 증가 → KV Cache N배
- Quantization: FP16→INT8로 50% 절감

### 필수 최적화
실제 서비스에서는 다음이 필수적입니다:
- vLLM (Paged Attention)
- Quantization (FP8/INT8)
- 적절한 Context length 제한

---

## 8. 다음 학습 주제

KV Cache를 이해했다면 다음 주제들을 학습하세요:

1. **vLLM의 PagedAttention**
   - 메모리 효율 2~3배 향상 원리
   - Virtual memory 기반 KV 관리

2. **Multi-GPU KV Cache 분산**
   - Tensor Parallelism
   - Pipeline Parallelism

3. **긴 문맥 처리 최적화**
   - Attention의 Quadratic complexity 문제
   - Sparse Attention, Flash Attention

4. **Inference 서버 설계**
   - Request batching 전략
   - Memory pool 관리
   - Load balancing

5. **고급 Quantization**
   - GPTQ, AWQ
   - Mixed-precision inference
