---
layout: post
title: Agent 시스템 (Memory)
summary: agent 시스템의 주요 요소
author: keonhee
date: 2025-04-02 12:00:00 +0900
category: Agent system
keywords: Agent
permalink: /blog/agent_system_memory/
usemathjax: false
thumbnail: /assets/img/posts/agent_system_tree.png
imageNameKey: Agent System
---
![](assets/img/posts/agent_system_memory.png)
# Memory

---
### 1. Memory란?

Memory는 Agent가 과거의 경험, 수집한 정보, 현재 작업의 맥락을 **저장하고 필요할 때 꺼내 쓰는 능력**임. 인간으로 치면 단기 기억과 장기 기억에 해당하며, 단발성 응답에 그치는 LLM과 달리 Agent가 **상태를 유지하고 지속적으로 동작**할 수 있게 하는 핵심 요소임.

Memory가 없는 Agent는 매 순간 백지 상태에서 판단을 내려야 하므로, 복잡한 다단계 작업이나 장기적인 목표 달성이 불가능함. 즉, **Memory의 품질이 Agent의 지속성과 일관성을 결정**함.

Agent의 Memory는 저장 기간과 용도에 따라 아래 네 가지로 분류됨.

|유형|역할|
|---|---|
|Short-term Memory|현재 작업의 맥락과 중간 결과를 일시적으로 유지|
|Long-term Memory|외부 저장소에 정보를 영구적으로 보관하고 검색|
|Episodic Memory|과거 행동과 결과를 경험 단위로 기록|
|Semantic Memory|개념과 관계를 구조화된 지식으로 저장|
|Procedural Memory|반복적인 행동 절차와 도구 사용 패턴을 저장|

---

### 2. Short-term Memory

Short-term Memory는 Agent가 **현재 진행 중인 작업의 맥락과 중간 결과를 일시적으로 유지**하는 메모리임. 작업이 종료되면 사라지는 휘발성 메모리로, 인간의 단기 기억에 해당함. 현재 대화 내용, 중간 추론 결과, 임시 변수 등이 여기에 해당함.

#### 2-1. Context Window

Context Window는 LLM이 한 번에 처리할 수 있는 **최대 입력 길이** 안에서 유지되는 메모리임. Short-term Memory의 가장 기본적인 형태로, 현재 대화 내용과 작업 맥락이 여기에 저장됨.

- **Token Limit** : Context Window는 토큰 수로 제한되며, 이를 초과하면 오래된 정보부터 손실됨. 이를 관리하기 위한 전략이 필요함.
    - **Truncation Strategy** : 토큰 한계 초과 시 오래되거나 중요도가 낮은 정보를 잘라내는 방식. 어떤 정보를 우선적으로 제거할지 기준을 설정하는 것이 핵심임.
    - **Priority Scoring** : 저장된 정보에 중요도 점수를 부여하여, 한계 초과 시 낮은 점수의 정보부터 제거하는 방식임.
- **Window Management** : Context Window 내의 정보를 효율적으로 유지하기 위한 관리 전략임.
    - **Sliding Window** : 가장 최근 N개의 토큰만 유지하며 오래된 정보를 순차적으로 제거하는 방식임. 구현이 단순하나 초반 맥락이 손실될 수 있음.
    - **Compression** : 오래된 대화나 맥락을 요약하여 토큰 수를 줄이면서도 핵심 정보를 보존하는 방식임.

#### 2-2. Working Memory

Working Memory는 Agent가 현재 작업을 수행하는 동안 **중간 결과와 상태를 임시로 저장**하는 메모리임. 단순히 대화 맥락을 유지하는 Context Window와 달리, 추론 과정에서 생성되는 중간 산출물을 다룸.

- **Temporary Storage** : 작업 수행 중 생성되는 중간 결과를 임시로 저장하는 공간임.
    - **Variable Binding** : 추론 과정에서 생성된 중간 값을 변수에 바인딩하여 이후 단계에서 재사용할 수 있게 함.
    - **State Tracking** : 작업의 현재 진행 상태를 추적하여, 다음 단계에서 어디서부터 이어갈지 판단할 수 있게 함.
- **Attention Mechanism** : Working Memory 내에서 현재 처리 중인 정보가 저장된 정보 중 **어느 부분에 집중할지** 결정하는 메커니즘임.
    - **Self-attention** : 현재 입력 내의 토큰들이 서로 간의 연관성을 계산하여 중요한 부분에 집중함.
    - **Cross-attention** : 현재 처리 중인 정보가 Working Memory에 저장된 다른 정보와의 연관성을 계산하여 필요한 정보를 선택적으로 참조함.
---

#### 3. Preprocessing

입력 데이터는 LLM이 바로 처리할 수 있는 형태가 아닌 경우가 많음. Preprocessing은 다양한 형태의 입력을 **LLM이 처리 가능한 형태로 변환하는 과정**으로, Perception 단계에서 핵심적인 역할을 담당함.

##### 3-1. Tokenization

텍스트를 LLM이 처리할 수 있는 최소 단위인 **토큰(Token)** 으로 분할하는 과정임. 어떤 토크나이저를 사용하느냐에 따라 동일한 텍스트도 다르게 분할될 수 있음.

- **BPE (Byte Pair Encoding)** : 자주 등장하는 문자 쌍을 반복적으로 병합하여 어휘를 구축하는 방식임. GPT 계열 모델에서 주로 사용됨.
    - **Vocab Building** : 초기 문자 단위에서 시작해 자주 등장하는 쌍을 반복 병합하여 어휘 사전을 구축함.
    - **Merge Rules** : 병합 순서를 규칙으로 저장하여 새로운 텍스트에도 동일한 방식으로 토크나이징을 적용함.
- **WordPiece** : BPE와 유사하나 병합 기준이 다름. 학습 데이터에 없는 단어(OOV, Out-of-Vocabulary) 문제를 효과적으로 처리함. BERT 계열 모델에서 주로 사용됨.
    - **Subword Split** : 단어를 서브워드 단위로 분할하여 미등록 단어도 처리할 수 있게 함.
    - **OOV Handling** : 학습 데이터에 없는 단어를 서브워드 조합으로 표현하여 처리함.

##### 3-2. Embedding

텍스트를 **숫자 벡터로 변환**하여 의미적 유사도를 계산할 수 있게 하는 과정임. LLM이 언어를 수치적으로 이해하기 위한 필수 과정임.

- **Word Embedding** : 단어 단위로 고정된 벡터를 생성함. 문맥을 고려하지 못한다는 한계로 현재는 대부분 Contextual Embedding으로 대체됨.
    - **Word2Vec** : 주변 단어를 예측하는 방식으로 단어 벡터를 학습함. 의미적으로 유사한 단어는 벡터 공간에서 가깝게 위치함.
    - **GloVe** : 단어 간 동시 출현 빈도를 기반으로 벡터를 학습함. Word2Vec보다 전역적인 통계 정보를 반영함.
- **Contextual Embedding** : 동일한 단어라도 문맥에 따라 다른 벡터를 생성함. Agent 시스템에서는 주로 이 방식을 사용함.
    - **BERT** : 문장 전체의 양방향 문맥을 반영하여 벡터를 생성함. 문서 이해 태스크에 강점이 있음.
    - **Sentence Transformer** : 문장 단위의 벡터를 효율적으로 생성함. 문장 간 유사도 비교에 특화되어 RAG 시스템에서 널리 사용됨.

##### 3-3. Context Extraction

긴 문서에서 필요한 정보를 추출하고, context window 한계를 극복하기 위해 문서를 적절한 단위로 분할하는 과정임.

- **Sliding Window** : 일정 크기의 윈도우를 순차적으로 이동하며 텍스트를 분할함. Window Size(윈도우 크기)와 Overlap Strategy(중복 허용 범위)를 어떻게 설정하느냐가 성능에 직접적인 영향을 미침.
    - **Non-overlapping** : 윈도우가 겹치지 않고 순차적으로 이동함. 처리 속도가 빠르나 경계에서 문맥이 단절될 수 있음.
    - **Overlapping** : 윈도우가 일정 비율 겹치며 이동함. 경계 문맥 손실을 줄일 수 있으나 중복 처리로 비용이 증가함.
- **Chunking** : 문서를 의미 있는 단위로 분할함. 일반적으로 Semantic Chunking이 더 높은 품질의 결과를 제공함.
    - **Fixed-size Chunking** : 토큰 수 기준으로 고정된 크기로 분할함. 구현이 단순하나 문장 중간에서 잘릴 수 있음.
    - **Sentence-based Chunking** : 문장 단위로 분할함. 문장 경계를 보존하여 문맥 단절을 줄임.
    - **Paragraph-based Chunking** : 문단 단위로 분할함. 하나의 주제를 온전히 보존할 수 있음.
    - **Semantic Chunking** : 의미적 유사도를 기준으로 분할함. 가장 높은 품질을 제공하나 처리 비용이 높음.
    - **Recursive Chunking** : 분할 기준을 계층적으로 적용함. 문단 → 문장 → 단어 순으로 재귀적으로 분할함.

---

#### 4. Multimodal Processing

서로 다른 유형의 입력(텍스트 + 이미지 등)을 **통합적으로 처리하는 과정**임. 단일 모달리티만 처리하던 초기 AI 시스템과 달리, 최근 Agent 시스템은 복수의 입력 유형을 동시에 이해하고 활용하는 방향으로 발전하고 있음.

##### 4-1. Cross-modal Fusion

서로 다른 모달리티의 정보를 **하나로 합치는 방식**임. 언제 합치느냐에 따라 두 가지로 구분됨.

- **Early Fusion** : 입력 단계에서 먼저 합치는 방식. 모달리티 간 상호작용을 초기부터 반영할 수 있으나, 입력 형태가 다를 경우 정보 손실이 발생할 수 있음.
    - **Feature Concat** : 서로 다른 모달리티의 특징 벡터를 하나로 이어붙여 통합함.
    - **Joint Embedding** : 서로 다른 모달리티를 동일한 벡터 공간에 함께 매핑하여 통합함.
- **Late Fusion** : 각 모달리티를 독립적으로 처리한 후 결과를 합치는 방식. 각 모달리티의 특성을 독립적으로 보존할 수 있으나, 모달리티 간 상호작용을 반영하기 어려움.
    - **Score Averaging** : 각 모달리티의 처리 결과 점수를 평균내어 최종 결과를 도출함.
    - **Decision Voting** : 각 모달리티의 결과를 다수결로 취합하여 최종 결과를 결정함.

##### 4-2. Modality Alignment

서로 다른 모달리티 간의 **의미적 관계를 학습**하는 과정임. 예를 들어 "고양이 이미지"와 "고양이"라는 텍스트가 같은 의미임을 이해하는 것이 목표임.

- **Contrastive Learning** : 의미적으로 유사한 쌍은 벡터 공간에서 가깝게, 다른 쌍은 멀게 학습하는 방식임.
    - **CLIP-style** : 이미지와 텍스트를 동일한 벡터 공간에 정렬하여 서로 대응되는 쌍을 학습함.
    - **Triplet Loss** : Anchor·Positive·Negative 세 쌍을 비교하여 유사한 쌍은 가깝게, 다른 쌍은 멀게 학습함.
- **Cross-attention** : 서로 다른 모달리티 간의 상호작용을 Attention 메커니즘으로 학습하는 방식임.
    - **Query-Key Matching** : 한 모달리티의 Query와 다른 모달리티의 Key를 매칭하여 연관성을 계산함.
    - **Attention Weight** : Query-Key 매칭 결과를 기반으로 어느 부분에 집중할지를 가중치로 결정함.

---

#### 5. Context Window 관리

##### 5-1. Context Window란?

Context Window란 LLM이 한 번에 처리할 수 있는 **최대 입력 길이**를 의미함. 토큰 수로 측정되며, 이 범위를 초과하는 입력은 처리되지 않거나 초기 맥락이 손실됨. Perception 단계에서 가장 중요한 실무적 과제 중 하나임.

##### 5-2. Context Window 초과 시 발생하는 문제

- **맥락 손실** : 입력이 길어질수록 초반부 정보가 잘려나가 중요한 맥락을 잃을 수 있음.
- **성능 저하** : Context Window 한계에 가까워질수록 LLM의 추론 품질이 저하되는 경향이 있음.
- **비용 증가** : 처리하는 토큰 수가 많아질수록 API 호출 비용이 증가함.

##### 5-3. 관리 전략

- **Chunking** : 3-3에서 다룬 것처럼 긴 문서를 적절한 단위로 분할하여 필요한 부분만 입력으로 사용함.
- **핵심 정보 선별** : 전체 입력을 그대로 넣는 것이 아니라, 목표 달성에 필요한 정보만 추려서 입력함.
- **요약(Summarization)** : 긴 맥락을 압축하여 핵심만 유지한 채 토큰 수를 줄임.
- **외부 메모리 활용** : Context Window를 초과하는 정보는 외부 DB에 저장하고 필요할 때 검색하여 가져옴. 이는 Memory 단계와 직접 연결됨.