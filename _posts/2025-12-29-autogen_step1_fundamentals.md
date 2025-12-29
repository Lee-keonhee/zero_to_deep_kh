---
layout: post
title: AutoGen 완전 정복 Step 1 - 기초 개념과 ConversableAgent
summary: AutoGen의 핵심 개념, 대화 기반 multi-agent framework의 이해, orchestration과의 차이점
author: keonhee
date: 2025-12-29 15:00:00 +0900
category: AI Agent
keywords: AutoGen, Multi-Agent, ConversableAgent, LLM, Agent Framework, Orchestration
permalink: /blog/autogen_step1_fundamentals/
usemathjax: false
thumbnail: /assets/img/posts/autogen.png
imageNameKey: autogen
---

# AutoGen Step 1: 기초 개념과 ConversableAgent

## 목차
1. [AutoGen이란 무엇인가](#1-autogen이란-무엇인가)
2. [핵심 개념: ConversableAgent](#2-핵심-개념-conversableagent)
3. [Orchestration vs AutoGen](#3-orchestration-vs-autogen)
4. [적용 시나리오](#4-적용-시나리오)

---

## 1. AutoGen이란 무엇인가

### 개요

AutoGen은 Microsoft Research에서 개발한 **대화 기반 multi-agent framework**입니다. 
일반적인 LLM 호출이나 전통적인 orchestration과는 다른 접근 방식을 사용합니다.

### 주요 특징

```
전통적인 방식:
User → LLM → Response (1회성)

AutoGen 방식:
Agent A ↔ Agent B ↔ Agent C
(대화가 종료 조건을 만날 때까지 계속 상호작용)
```

### 핵심 차별점

1. **대화 중심(Conversational)**: Agent들이 메시지를 주고받으며 문제 해결
2. **자율성(Autonomy)**: 미리 정의된 플로우 없이 상황에 따라 적응
3. **유연성(Flexibility)**: 중간 결과에 따라 전략 변경 가능

---

## 2. 핵심 개념: ConversableAgent

### 기본 구조

AutoGen의 모든 agent는 `ConversableAgent` 클래스를 기반으로 합니다.

```python
from autogen import ConversableAgent

# LLM 기반 Agent
assistant = ConversableAgent(
    name="Assistant",
    llm_config={"model": "gpt-4", "api_key": "your-api-key"},
)

# Human proxy 또는 코드 실행 Agent
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,  # LLM 사용 안함
    human_input_mode="ALWAYS",  # 항상 사람 입력 받음
)
```

### Agent의 역할

**LLM Agent (assistant)**
- LLM을 사용하여 응답 생성
- 문제 분석, 코드 작성, 설명 제공

**Human Proxy (user_proxy)**
- 사람의 입력을 대신하거나
- 코드 실행, 도구 사용 담당
- LLM 없이 작동 가능

### 대화 시작하기

```python
# 대화 초기화
user_proxy.initiate_chat(
    assistant,
    message="Hello! Tell me a joke."
)
```

**중요**: `initiate_chat()`는 단순 메시지 전송이 아닙니다!

### 대화 흐름 예시

```python
# 실제 대화 플로우
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci(10) and execute it."
)

"""
실제 진행:
Turn 1: user_proxy → assistant
"Write a Python function to calculate fibonacci(10) and execute it."

Turn 2: assistant → user_proxy
"Sure! Here's the code:
```python
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)
print(fib(10))
```"

Turn 3: user_proxy → assistant
[자동으로 코드 실행]
"Code execution result: 55"

Turn 4: assistant → user_proxy
"Great! The 10th Fibonacci number is 55. TERMINATE"
"""
```

### 대화 종료 조건

대화는 다음 조건 중 하나를 만족할 때 종료됩니다:

```python
# 1. max_consecutive_auto_reply 도달
assistant = ConversableAgent(
    name="Assistant",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=5  # 최대 5번 자동 응답
)

# 2. 특정 키워드 감지
user_proxy.initiate_chat(
    assistant,
    message="Complete this task.",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 3. 사람 개입
human_proxy = ConversableAgent(
    name="Human",
    llm_config=False,
    human_input_mode="ALWAYS"  # 매 턴마다 사람이 개입
)
```

### 코드 실행 설정

```python
# 코드 자동 실행 Agent
code_executor = ConversableAgent(
    name="CodeExecutor",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",  # 작업 디렉토리
        "use_docker": False     # Docker 사용 여부
    }
)
```

---

## 3. Orchestration vs AutoGen

### 전통적인 Orchestration

```python
# 중앙 컨트롤러가 모든 흐름 관리
class Orchestrator:
    def process_task(self, task):
        # Step 1: 코드 작성
        code = code_writer_agent.write(task)
        
        # Step 2: 코드 실행
        result = code_executor_agent.execute(code)
        
        # Step 3: 결과 분석
        analysis = analyzer_agent.analyze(result)
        
        return analysis

# 장점: 예측 가능, 제어 용이
# 단점: 유연성 부족, 사전 정의 필요
```

### AutoGen 방식

```python
# Agent들이 대화로 협업
assistant.initiate_chat(
    user_proxy,
    message="Analyze this dataset and find insights."
)

"""
실제 대화:
Assistant: "I'll start by loading the data..."
UserProxy: [코드 실행] "Data loaded. 1000 rows, 5 columns"
Assistant: "I see missing values. Let me clean them..."
UserProxy: [코드 실행] "Cleaned. 950 rows remain"
Assistant: "Now let's look at correlations..."
... (대화 계속)
Assistant: "Found 3 key insights. TERMINATE"
"""

# 장점: 유연성, 적응성, 탐색적 문제 해결
# 단점: 예측 어려움, 비용 증가 가능
```

### 비교표

| 특징 | Orchestration | AutoGen |
|------|--------------|---------|
| 제어 방식 | 중앙 집중형 | 분산형 (대화) |
| 플로우 | 미리 정의 | 동적 생성 |
| 적응성 | 낮음 | 높음 |
| 예측성 | 높음 | 낮음 |
| 적용 사례 | 데이터 파이프라인 | 탐색적 분석, 코딩 어시스턴트 |

---

## 4. 적용 시나리오

### 시나리오 A: 정형화된 작업 (Orchestration 적합)

```python
"""
작업: CSV 파일 처리
1. 파일 읽기
2. 데이터 클리닝 (고정 규칙)
3. 통계 계산
4. 리포트 생성

→ 단계가 명확하고 변하지 않음
→ Orchestration이 효율적
"""
```

### 시나리오 B: 탐색적 작업 (AutoGen 적합)

```python
"""
작업: "이 데이터에서 흥미로운 인사이트를 찾아줘"

→ 무엇을 먼저 봐야 할지 모름
→ 중간 발견에 따라 분석 방향 변경
→ 에러 발생 시 다른 접근법 시도

→ AutoGen의 유연성 필요
"""

# AutoGen 예시
assistant.initiate_chat(
    user_proxy,
    message="Explore this sales data and find actionable insights."
)

"""
가능한 대화 흐름:
1. Assistant: 먼저 기본 통계 확인
2. UserProxy: [실행] 특정 제품 매출이 급증
3. Assistant: 해당 제품 시계열 분석
4. UserProxy: [실행] 특정 달에만 급증
5. Assistant: 프로모션 이력 확인 필요
... (동적으로 계속)
"""
```

### 시나리오 C: 코딩 어시스턴트

```python
# AutoGen이 강력한 사례
coder = ConversableAgent(
    name="Coder",
    llm_config={"model": "gpt-4"},
)

executor = ConversableAgent(
    name="Executor",
    llm_config=False,
    code_execution_config={"work_dir": "workspace"}
)

coder.initiate_chat(
    executor,
    message="Create a web scraper for news headlines and save to CSV."
)

"""
대화 진행:
1. Coder: 라이브러리 설치 필요 (requests, beautifulsoup4)
2. Executor: [실행] pip install ...
3. Coder: 스크래핑 코드 작성
4. Executor: [실행] ImportError 발생
5. Coder: 코드 수정
6. Executor: [실행] 성공! data.csv 생성
7. Coder: TERMINATE
"""
```

### 실전 예제: 데이터 분석

```python
from autogen import ConversableAgent

# 분석가 Agent
analyst = ConversableAgent(
    name="DataAnalyst",
    system_message="You are a data analyst. Write Python code to analyze data.",
    llm_config={
        "model": "gpt-4",
        "api_key": "your-key",
        "temperature": 0
    }
)

# 실행 Agent
executor = ConversableAgent(
    name="Executor",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "analysis",
        "use_docker": False
    }
)

# 대화 시작
executor.initiate_chat(
    analyst,
    message="""
    I have a CSV file 'sales_data.csv' with columns: date, product, quantity, revenue.
    Please analyze it and provide:
    1. Top 5 products by revenue
    2. Monthly revenue trend
    3. Any interesting patterns
    """
)
```

---

## 핵심 정리

### AutoGen의 본질

```
1. 대화 기반: Agent들이 메시지를 주고받으며 협업
2. 자율성: 중앙 컨트롤러 없이 자율적으로 문제 해결
3. 적응성: 미리 정의된 플로우 없이 상황에 따라 변화
```

### 언제 사용할까?

**AutoGen 사용:**
- 탐색적 문제 해결
- 복잡하고 예측 불가능한 작업
- 코딩 어시스턴트
- 대화형 AI 애플리케이션

**Orchestration 사용:**
- 명확한 워크플로우
- 프로덕션 파이프라인
- 비용 효율성 중요
- 예측 가능성 필요

### 주요 개념 체크리스트

✅ ConversableAgent = AutoGen의 기본 빌딩 블록
✅ 대화는 종료 조건을 만날 때까지 계속됨
✅ LLM Agent와 실행 Agent의 역할 분리
✅ Orchestration과의 근본적 차이 이해
✅ 적절한 사용 시나리오 판단 능력

---

## 다음 단계

Step 1을 완료했습니다! 이제 다음을 이해했습니다:

✅ AutoGen의 핵심 개념
✅ ConversableAgent의 기본 구조
✅ 대화 기반 협업 방식
✅ Orchestration과의 차이점

**다음 학습 옵션**:

**Step 2A - Reply 메커니즘**
- Agent가 언제/어떻게 응답하는지
- 종료 조건 상세 설정
- 자동 응답 커스터마이징

**Step 2B - 실전 패턴**
- Two-agent collaboration 심화
- 코드 생성 + 실행 패턴
- 실제 프로젝트 예제

**Step 2C - Multi-agent 구조**
- 3개 이상 agent 협업
- Group Chat 패턴
- Agent 간 역할 분담

어떤 단계로 진행하시겠습니까?
