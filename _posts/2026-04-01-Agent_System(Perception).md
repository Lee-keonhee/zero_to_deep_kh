---
layout: post
title: Agent 시스템 (Perception)
summary: agent 시스템의 주요 요소
author: keonhee
date: 2025-04-01 12:00:00 +0900
category: Agent system
keywords: Agent
permalink: /blog/agent_system_perception/
usemathjax: false
thumbnail: /assets/img/posts/agent_system_tree.png
imageNameKey: Agent System
---
![](assets/img/posts/agent_system_perception.png)
# Perception

---
#### 1. Perception이란?

Perception은 Agent가 외부 환경으로부터 입력을 받아 **이해 가능한 형태로 변환하는 과정**임. 인간으로 치면 "보고 듣고 읽는" 감각 기관에 해당하며, 이후 Memory·Planning 등 모든 단계의 출발점이 됨.

아무리 뛰어난 추론 능력을 갖춘 Agent라도 입력을 제대로 인식하지 못하면 올바른 판단을 내릴 수 없음. 즉, **Perception의 품질이 Agent 전체 성능의 상한선을 결정**한다고 볼 수 있음.

Perception 단계에서 Agent가 수행하는 핵심 역할은 다음과 같음.

- **입력 수신** : 사용자 요청, 파일, API 응답 등 다양한 형태의 외부 입력을 받아들임.
- **형태 변환** : 수신한 입력을 LLM이 처리할 수 있는 형태로 변환함.
- **맥락 추출** : 입력에서 목표 달성에 필요한 핵심 정보를 추출함.

---

#### 2. Input Types

Agent가 처리할 수 있는 입력 유형은 크게 **Text, Image, Audio** 세 가지로 분류됨. 각 유형은 처리 방식과 활용 목적이 상이하므로, Agent 설계 시 어떤 입력을 다룰지 명확히 정의하는 것이 중요함.

##### 2-1. Text

가장 기본적이고 널리 사용되는 입력 유형으로, 구조화 여부에 따라 두 가지로 구분됨.

- **Structured** : 명확한 형식을 가진 텍스트. JSON, XML, Table, CSV 등이 해당하며, 형식이 정해져 있어 파싱이 용이함.
- **Unstructured** : 형식이 정해지지 않은 텍스트. 자연어, 코드 등이 해당하며, 의미 파악을 위한 별도의 처리가 필요함.

##### 2-2. Image

시각적 정보를 담은 입력 유형으로, 표현 방식에 따라 두 가지로 구분됨.

- **Raster** : 픽셀 기반 이미지. PNG, JPEG, 스크린샷 등이 해당하며, 해상도에 따라 품질이 결정됨.
- **Vector** : 수학적 좌표 기반 이미지. SVG, 다이어그램 등이 해당하며, 해상도에 독립적으로 품질이 유지됨.

##### 2-3. Audio

소리 형태의 입력 유형으로, 음성 포함 여부에 따라 두 가지로 구분됨.

- **Speech** : 사람의 음성 입력. STT(Speech-to-Text)로 텍스트 변환 후 처리하거나, 화자 식별(Speaker ID)에 활용됨.
- **Non-speech** : 음성 외의 소리 입력. 환경음 분류(Sound Classification), 노이즈 제거(Noise Filtering) 등에 활용됨.

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