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

[예시 코드 보러 가기](2026-04-02-Agent_System(Memory)_code)
#### 2-2. Working Memory

Working Memory는 Agent가 현재 작업을 수행하는 동안 **중간 결과와 상태를 임시로 저장**하는 메모리임. 단순히 대화 맥락을 유지하는 Context Window와 달리, 추론 과정에서 생성되는 중간 산출물을 다룸.

- **Temporary Storage** : 작업 수행 중 생성되는 중간 결과를 임시로 저장하는 공간임.
    - **Variable Binding** : 추론 과정에서 생성된 중간 값을 변수에 바인딩하여 이후 단계에서 재사용할 수 있게 함.
    - **State Tracking** : 작업의 현재 진행 상태를 추적하여, 다음 단계에서 어디서부터 이어갈지 판단할 수 있게 함.
    
	실제 구현 방식은 목적에 따라 두 가지로 나뉨.

| 방식               | 설명                                                                       | 적합한 상황                 |
| ---------------- | ------------------------------------------------------------------------ | ---------------------- |
| **Scratchpad**   | LLM이 텍스트로 중간 과정을 직접 써내려가는 방식. Claude의 thinking 태그, Chain-of-Thought가 대표적 | 추론 흐름이 동적으로 바뀌는 작업     |
| **State Schema** | 타입을 미리 정의하고 노드 간에 상태를 넘기는 방식. LangGraph가 대표적. Redis/DB 같은 외부 저장소와 조합 가능  | 작업 구조가 미리 정해진 멀티 스텝 작업 |

- **Attention Mechanism** : Working Memory 내에서 현재 처리 중인 정보가 저장된 정보 중 **어느 부분에 집중할지** 결정하는 메커니즘임.
    - **Self-attention** : 현재 입력 내의 토큰들이 서로 간의 연관성을 계산하여 중요한 부분에 집중함.
    - **Cross-attention** : 현재 처리 중인 정보가 Working Memory에 저장된 다른 정보와의 연관성을 계산하여 필요한 정보를 선택적으로 참조함.

---

### 3. Long-term Memory

Long-term Memory는 Agent가 **작업 종료 후에도 정보를 영구적으로 보관하고, 필요할 때 검색하여 활용**하는 메모리임. Short-term Memory가 휘발성인 것과 달리, 외부 저장소에 정보를 저장하여 세션이 종료되어도 유지됨. 대규모 문서, 과거 대화 이력, 도메인 지식 등이 여기에 해당함.

#### 3-1. Vector DB

Vector DB는 텍스트·이미지 등의 데이터를 **벡터로 변환하여 저장하고, 의미적 유사도 기반으로 검색**할 수 있는 데이터베이스임. Long-term Memory의 핵심 저장소로, RAG 시스템의 기반이 됨.

- **Indexing** : 벡터를 효율적으로 검색하기 위해 색인을 구축하는 과정임.
    - **HNSW (Hierarchical Navigable Small World)** : 계층적 그래프 구조로 벡터를 색인하는 방식. 검색 속도가 빠르고 정확도가 높아 현재 가장 널리 사용됨.
    - **IVF (Inverted File Index)** : 벡터 공간을 클러스터로 나누어 색인하는 방식. 대규모 데이터셋에서 메모리 효율이 높음.
- **Query** : 저장된 벡터 중 입력 벡터와 가장 유사한 항목을 찾는 과정임.
    - **ANN Search (Approximate Nearest Neighbor)** : 정확한 최근접 이웃을 찾는 대신, 근사치를 빠르게 탐색하는 방식. 속도와 정확도 간의 트레이드오프가 존재함.
    - **Similarity Metric** : 벡터 간 유사도를 측정하는 기준. Cosine Similarity, Euclidean Distance, Dot Product 등이 사용됨.

#### 3-2. RAG (Retrieval-Augmented Generation)

RAG는 LLM이 응답을 생성할 때 **외부 저장소에서 관련 정보를 검색하여 함께 활용**하는 방식임. LLM의 학습 데이터 한계를 극복하고, 최신 정보나 도메인 특화 지식을 반영할 수 있게 함.

- **Retrieval** : 쿼리와 관련된 정보를 외부 저장소에서 검색하는 과정임.
    - **Dense Retrieval** : 쿼리와 문서를 모두 벡터로 변환하여 의미적 유사도로 검색하는 방식. 문맥적 의미를 반영할 수 있으나 계산 비용이 높음.
    - **Sparse Retrieval** : TF-IDF, BM25 등 키워드 기반으로 검색하는 방식. 계산 비용이 낮고 특정 키워드 검색에 강점이 있으나 의미적 유사도를 반영하지 못함.
- **Augmentation** : 검색된 정보를 LLM의 입력에 통합하여 응답 품질을 높이는 과정임.
    - **Context Injection** : 검색된 문서를 LLM의 프롬프트에 직접 삽입하여 응답 생성에 활용함.
    - **Reranking** : 검색된 문서들을 관련성 기준으로 재정렬하여 가장 유용한 정보를 우선적으로 활용함.

#### 3-3. Retrieval Strategy

검색의 품질과 효율을 높이기 위한 전략임. 어떤 방식으로 검색하느냐에 따라 Agent의 응답 품질이 크게 달라짐.

- **Top-K Search** : 유사도 점수가 높은 상위 K개의 결과를 반환하는 방식임.
    - **Score Threshold** : 일정 점수 이상의 결과만 반환하여 낮은 관련성의 정보가 포함되는 것을 방지함.
    - **Diversity Filter** : 유사한 내용의 문서가 중복으로 반환되지 않도록 다양성을 확보하는 필터임.
- **Hybrid Search** : Dense Retrieval과 Sparse Retrieval을 결합하여 각각의 단점을 보완하는 방식임.
    - **BM25 + Dense** : 키워드 기반 BM25와 의미 기반 Dense Retrieval을 함께 사용하여 검색 커버리지를 높임.
    - **Score Fusion** : 두 방식의 검색 결과 점수를 통합하여 최종 순위를 결정하는 방식. RRF(Reciprocal Rank Fusion) 등이 대표적임.

---
### 4. Episodic Memory

Episodic Memory는 Agent가 **과거에 수행한 작업의 경험을 기록하고, 이를 이후 판단에 활용**하는 메모리임. 단순히 정보를 저장하는 Long-term Memory와 달리, "언제, 무엇을, 어떻게 했고, 결과가 어땠는지"를 **경험 단위로 구조화하여 기록**함. 이를 통해 Agent는 유사한 상황에서 과거 경험을 참고해 더 나은 판단을 내릴 수 있음.

#### 4-1. Experience Log

Agent가 수행한 작업의 **행동과 결과를 기록**하는 공간임.

- **Action History** : Agent가 수행한 행동의 이력을 순서대로 기록함.
    - **Timestamping** : 각 행동이 언제 수행되었는지 시간 정보를 함께 기록하여 시간적 맥락을 보존함.
    - **Action Tagging** : 각 행동에 유형·목적 등의 태그를 부여하여 이후 검색과 분류를 용이하게 함.
- **Outcome Recording** : 각 행동의 결과를 기록하여 성공·실패 여부를 추적함.
    - **Success / Fail** : 작업의 성공 또는 실패 여부를 명시적으로 기록함. 이후 유사한 상황에서 동일한 실수를 반복하지 않도록 하는 데 활용됨.
    - **Reward Signal** : 행동의 결과에 대한 보상 점수를 기록함. 어떤 행동이 목표 달성에 얼마나 기여했는지를 수치화하여 이후 의사결정에 반영함.

#### 4-2. Event Indexing

기록된 경험을 **효율적으로 검색하고 활용**하기 위해 색인을 구축하는 과정임.

- **Temporal Index** : 시간 기반으로 경험을 색인하는 방식임.
    - **Time-based Query** : 특정 시간대에 수행된 작업을 검색할 수 있게 함. "어제 수행한 작업", "지난주에 실패한 작업" 등의 조회가 가능함.
    - **Recency Weighting** : 최근 경험일수록 높은 가중치를 부여하여 검색 결과에 우선적으로 반영함. 오래된 경험보다 최근 경험이 현재 상황에 더 관련성이 높다는 가정에 기반함.
- **Semantic Index** : 의미 기반으로 경험을 색인하는 방식임.
    - **Event Embedding** : 경험을 벡터로 변환하여 의미적으로 유사한 경험을 검색할 수 있게 함. 현재 상황과 유사했던 과거 경험을 효율적으로 찾는 데 활용됨.
    - **Cluster Grouping** : 유사한 경험들을 클러스터로 묶어 관리함. 비슷한 유형의 작업 패턴을 파악하고 반복적인 실패나 성공 패턴을 분석하는 데 활용됨.

---

