---
layout: post
title: AutoGen 완전 정복 Step 2 - Reply 메커니즘과 Agent 협업 패턴
summary: max_consecutive_auto_reply, 종료 조건 설정, Two-Agent 패턴, Sequential Workflow, GroupChat까지 완벽 마스터
author: keonhee
date: 2025-12-29 16:00:00 +0900
category: AI Agent
keywords: AutoGen, Multi-Agent, Reply Mechanism, GroupChat, Agent Collaboration, LLM Framework
permalink: /blog/autogen_step2_agent_collaboration/
usemathjax: false
thumbnail: /assets/img/posts/autogen.png
imageNameKey: autogen
---

# AutoGen Step 2: Reply 메커니즘과 Agent 협업 패턴

## 목차
1. [Reply 메커니즘 기초](#1-reply-메커니즘-기초)
2. [Two-Agent 패턴: 코딩 어시스턴트](#2-two-agent-패턴-코딩-어시스턴트)
3. [Sequential Workflow: 단계별 작업 흐름](#3-sequential-workflow-단계별-작업-흐름)
4. [GroupChat: 자유로운 협업](#4-groupchat-자유로운-협업)

---

## 1. Reply 메커니즘 기초

### max_consecutive_auto_reply - 자동 응답 횟수 제한

AutoGen의 agent는 대화가 끝날 때까지 자동으로 응답합니다. 
무한 루프를 방지하기 위해 `max_consecutive_auto_reply`로 최대 응답 횟수를 제한합니다.

```python
from autogen import ConversableAgent

# Agent A: 최대 10번 자동 응답
agent_a = ConversableAgent(
    name="AgentA",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=10
)

# Agent B: 최대 5번 자동 응답
agent_b = ConversableAgent(
    name="AgentB",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=5
)

# 대화 시작
agent_a.initiate_chat(agent_b, message="Hello!")
```

**동작 원리:**
```
Turn 1: A → B (A 카운트: 1)
Turn 2: B → A (B 카운트: 1)
Turn 3: A → B (A 카운트: 2)
Turn 4: B → A (B 카운트: 2)
...
Turn 9: A → B (A 카운트: 5)
Turn 10: B → A (B 카운트: 5) ← B의 한계
Turn 11: A는 응답하려 하지만, B가 더 이상 응답 못함 → 대화 종료
```

**중요:** 두 agent 중 **먼저 한계에 도달한 쪽**이 응답을 멈추면 대화가 종료됩니다.

### is_termination_msg - 종료 조건 설정

특정 메시지 내용을 감지하여 대화를 종료할 수 있습니다.

```python
# 기본 예제: "TERMINATE" 감지
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

assistant = ConversableAgent(
    name="Assistant",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

user_proxy.initiate_chat(
    assistant,
    message="Write a factorial function."
)
```

**핵심 개념:**
- `is_termination_msg`는 **받은 메시지**를 체크하는 함수입니다
- user_proxy가 assistant의 응답을 받았을 때 체크
- lambda 함수가 `True`를 반환하면 대화 종료

### 다양한 종료 조건 예제

```python
# 1. 특정 키워드 감지
is_termination_msg=lambda x: "완료" in x.get("content", "")

# 2. 여러 키워드 중 하나
is_termination_msg=lambda x: any(
    keyword in x.get("content", "") 
    for keyword in ["TERMINATE", "완료", "끝"]
)

# 3. 실행 성공 감지
is_termination_msg=lambda x: "exitcode: 0" in x.get("content", "")

# 4. 복잡한 조건
def check_termination(message):
    content = message.get("content", "")
    # 성공 메시지이거나 최종 결과가 있으면 종료
    return ("TERMINATE" in content) or ("최종 결과:" in content)

is_termination_msg=check_termination
```

### 적절한 max_consecutive_auto_reply 값 설정

```python
# 간단한 작업 (factorial 계산, 간단한 코드 작성)
max_consecutive_auto_reply=5

# 중간 복잡도 (데이터 분석, 웹 스크래핑)
max_consecutive_auto_reply=7

# 복잡한 작업 (디버깅 많이 필요, 여러 단계 작업)
max_consecutive_auto_reply=10

# 매우 복잡한 작업
max_consecutive_auto_reply=15
```

**가이드라인:**
- 성공 케이스: 보통 4~6턴
- 1번 에러 수정: 6~8턴
- 2번 에러 수정: 8~10턴
- 너무 크면 비용/시간 낭비, 무한 루프 위험

### 실전 예제: 기본 Reply 메커니즘

```python
from autogen import ConversableAgent

# LLM Agent
chatbot = ConversableAgent(
    name="챗봇",
    system_message="당신은 친절한 대화 상대입니다.",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=3
)

# Human proxy
user = ConversableAgent(
    name="사용자",
    llm_config=False,
    human_input_mode="ALWAYS",  # 매번 사람 입력
    is_termination_msg=lambda x: "종료" in x.get("content", "")
)

# 대화 시작
user.initiate_chat(
    chatbot,
    message="안녕하세요! 오늘 날씨 어때요?"
)

# 사용자가 "종료"라고 입력하면 대화 종료
```

---

## 2. Two-Agent 패턴: 코딩 어시스턴트

### 핵심 아이디어

코딩 어시스턴트는 두 개의 agent로 구성됩니다:
1. **Coder (LLM)**: 코드 작성 및 수정
2. **Executor (코드 실행)**: 코드 실행 및 결과 반환

```python
from autogen import ConversableAgent

# 코드 작성 Agent
coder = ConversableAgent(
    name="코더",
    system_message="""당신은 파이썬 프로그래머입니다. 깔끔하고 실행 가능한 코드를 작성하세요.
실행 결과를 받은 후 결과가 올바르면 'TERMINATE'로 응답하세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=7
)

# 코드 실행 Agent
executor = ConversableAgent(
    name="실행기",
    llm_config=False,  # LLM 사용 안함
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",  # 작업 디렉토리
        "use_docker": False
    },
    max_consecutive_auto_reply=7,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 대화 시작
executor.initiate_chat(
    coder,
    message="5의 팩토리얼을 계산하는 함수를 작성하고 실행해주세요."
)
```

### 대화 흐름 예시

```
Turn 1: Executor → Coder
"5의 팩토리얼을 계산하는 함수를 작성하고 실행해주세요."

Turn 2: Coder → Executor
"여기 코드입니다:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
print(factorial(5))
```"

Turn 3: Executor → Coder
[자동으로 코드 실행]
"exitcode: 0 (execution succeeded)
Code output: 
120"

Turn 4: Coder → Executor
"완벽합니다! 5의 팩토리얼은 120입니다. TERMINATE"

→ Executor가 "TERMINATE" 감지 → 대화 종료
```

### system_message 작성 가이드

**잘못된 예시:**
```python
# ❌ 모호한 지시
system_message="코드를 작성하고 완료되면 TERMINATE를 출력하세요."
# 문제: "TERMINATE를 출력"하면 print("TERMINATE")로 오해 가능

# ❌ 너무 짧음
system_message="파이썬 프로그래머"
# 문제: 종료 조건이 없음
```

**올바른 예시:**
```python
# ✅ 명확한 역할과 종료 조건
system_message="""당신은 파이썬 프로그래머입니다. 깔끔하고 실행 가능한 코드를 작성하세요.
실행 결과를 받은 후 결과가 올바르면 'TERMINATE'로 응답하세요."""

# ✅ 구체적인 행동 지시
system_message="""당신은 데이터 분석 전문가입니다.
1. 주어진 데이터를 분석하는 코드를 작성하세요
2. 실행 결과를 확인하고 올바르면 'TERMINATE'로 응답하세요
3. 에러가 발생하면 코드를 수정하세요"""

# ✅ 전문 영역 특화
system_message="""당신은 웹 스크래핑 전문가입니다.
BeautifulSoup과 requests를 사용하여 코드를 작성하세요.
데이터 수집이 완료되고 파일로 저장되면 'TERMINATE'로 응답하세요."""
```

### 실전 예제: 데이터 분석 어시스턴트

```python
from autogen import ConversableAgent

# 데이터 분석가
analyst = ConversableAgent(
    name="데이터분석가",
    system_message="""당신은 pandas를 사용하는 데이터 분석 전문가입니다.
1. 데이터를 로드하고 분석하는 코드를 작성하세요
2. 시각화가 필요하면 matplotlib를 사용하세요
3. 분석이 완료되고 결과를 확인하면 'TERMINATE'로 응답하세요""",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=10
)

# 실행기
executor = ConversableAgent(
    name="실행기",
    llm_config=False,
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "analysis",
        "use_docker": False
    },
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 분석 시작
executor.initiate_chat(
    analyst,
    message="""
    'sales_data.csv' 파일을 분석해주세요:
    1. 월별 매출 추이
    2. 상위 5개 제품
    3. 결과를 그래프로 저장
    """
)
```

### code_execution_config 옵션

```python
code_execution_config={
    "work_dir": "coding",        # 작업 디렉토리
    "use_docker": False,          # Docker 사용 여부
    "timeout": 60,                # 실행 제한 시간 (초)
    "last_n_messages": 1          # 마지막 n개 메시지에서 코드 추출
}

# Docker 사용 예시 (안전성 향상)
code_execution_config={
    "work_dir": "coding",
    "use_docker": True,
    "docker_image": "python:3.10"
}
```

---

## 3. Sequential Workflow: 단계별 작업 흐름

### 개념

복잡한 작업을 여러 단계로 나누어 순차적으로 처리하는 패턴입니다.
각 단계마다 전문화된 agent pair (coder + executor)를 사용합니다.

```
데이터 수집 → 데이터 정제 → 데이터 분석 → 리포트 생성
   (pair)        (pair)        (pair)        (pair)
```

### 기본 구조

```python
from autogen import ConversableAgent

# 1단계: 데이터 수집
collector_coder = ConversableAgent(
    name="수집_코더",
    system_message="당신은 데이터 수집 전문가입니다. 데이터를 수집하고 저장하는 코드를 작성하세요.",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=7
)

collector_executor = ConversableAgent(
    name="수집_실행기",
    llm_config=False,
    code_execution_config={"work_dir": "pipeline"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 2단계: 데이터 정제
cleaner_coder = ConversableAgent(
    name="정제_코더",
    system_message="당신은 데이터 정제 전문가입니다. 결측치, 중복 제거 등 데이터를 정제하세요.",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=7
)

cleaner_executor = ConversableAgent(
    name="정제_실행기",
    llm_config=False,
    code_execution_config={"work_dir": "pipeline"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 3단계: 데이터 분석
analyzer_coder = ConversableAgent(
    name="분석_코더",
    system_message="당신은 데이터 분석 전문가입니다. 통계 분석과 시각화를 수행하세요.",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=7
)

analyzer_executor = ConversableAgent(
    name="분석_실행기",
    llm_config=False,
    code_execution_config={"work_dir": "pipeline"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)
```

### Sequential 실행 패턴

```python
# 방법 1: 순차 실행
# Step 1: 데이터 수집
collector_executor.initiate_chat(
    collector_coder,
    message="sales_api에서 데이터를 수집하고 'raw_data.csv'로 저장하세요."
)

# Step 2: 데이터 정제
cleaner_executor.initiate_chat(
    cleaner_coder,
    message="""
    이전 단계에서 'raw_data.csv'가 생성되었습니다.
    이 파일을 정제하고 'cleaned_data.csv'로 저장하세요.
    """
)

# Step 3: 데이터 분석
analyzer_executor.initiate_chat(
    analyzer_coder,
    message="""
    'cleaned_data.csv'를 분석하세요:
    1. 월별 매출 추이
    2. 상위 제품 분석
    3. 결과를 'analysis.png'로 저장
    """
)
```

### 파일 기반 데이터 전달

**핵심:** 모든 agent가 같은 `work_dir`를 공유하면 파일로 데이터를 전달할 수 있습니다.

```python
# 모든 executor가 같은 디렉토리 사용
code_execution_config={"work_dir": "pipeline"}

# 단계별 파일 흐름
Step 1: raw_data.csv 생성
        ↓
Step 2: raw_data.csv 읽기 → cleaned_data.csv 생성
        ↓
Step 3: cleaned_data.csv 읽기 → analysis.png 생성
```

### 실전 예제: 웹 스크래핑 파이프라인

```python
from autogen import ConversableAgent

# 단계 1: 웹 스크래핑
scraper_coder = ConversableAgent(
    name="스크래퍼",
    system_message="""당신은 웹 스크래핑 전문가입니다.
BeautifulSoup과 requests를 사용하여 뉴스 헤드라인을 수집하세요.
결과를 'headlines.json'으로 저장하고 'TERMINATE'로 응답하세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

scraper_executor = ConversableAgent(
    name="스크래퍼_실행기",
    llm_config=False,
    code_execution_config={"work_dir": "news_pipeline"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 단계 2: 데이터 전처리
preprocessor_coder = ConversableAgent(
    name="전처리기",
    system_message="""당신은 텍스트 전처리 전문가입니다.
'headlines.json'을 읽어서 텍스트를 정제하고 'processed_headlines.json'으로 저장하세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

preprocessor_executor = ConversableAgent(
    name="전처리_실행기",
    llm_config=False,
    code_execution_config={"work_dir": "news_pipeline"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 단계 3: 감성 분석
sentiment_coder = ConversableAgent(
    name="감성분석기",
    system_message="""당신은 감성 분석 전문가입니다.
'processed_headlines.json'의 각 헤드라인에 대해 긍정/부정/중립을 분류하세요.
결과를 'sentiment_results.csv'로 저장하세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

sentiment_executor = ConversableAgent(
    name="감성분석_실행기",
    llm_config=False,
    code_execution_config={"work_dir": "news_pipeline"},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# 실행
print("=== 1단계: 웹 스크래핑 ===")
scraper_executor.initiate_chat(
    scraper_coder,
    message="https://news.example.com에서 최신 뉴스 헤드라인 50개를 수집하세요."
)

print("\n=== 2단계: 전처리 ===")
preprocessor_executor.initiate_chat(
    preprocessor_coder,
    message="'headlines.json' 파일을 전처리하세요."
)

print("\n=== 3단계: 감성 분석 ===")
sentiment_executor.initiate_chat(
    sentiment_coder,
    message="'processed_headlines.json'을 감성 분석하세요."
)
```

### Sequential vs GroupChat

**Sequential Workflow:**
```
장점:
- 명확한 단계별 흐름
- 디버깅 쉬움
- 각 단계 독립적 테스트 가능

단점:
- 유연성 부족
- 이전 단계로 돌아갈 수 없음
- 미리 정의된 순서만 가능
```

**언제 사용:**
- 명확한 단계가 있는 작업
- 데이터 파이프라인
- ETL 프로세스
- 보고서 생성 워크플로우

---

## 4. GroupChat: 자유로운 협업

### 핵심 개념

GroupChat은 여러 agent가 자유롭게 대화하며 협업하는 패턴입니다.
**GroupChatManager**가 대화 내용을 보고 다음 발언자를 선택합니다.

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Agent 생성
collector = ConversableAgent(
    name="데이터수집가",
    system_message="당신은 데이터 수집 전문가입니다. 데이터 출처나 수집 문제에 대해 답변하세요.",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

cleaner = ConversableAgent(
    name="데이터정제가",
    system_message="당신은 데이터 정제 전문가입니다. 이상한 패턴을 발견하면 데이터수집가에게 물어보세요.",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

analyzer = ConversableAgent(
    name="데이터분석가",
    system_message="당신은 데이터 분석 전문가입니다. 데이터가 이상하면 데이터정제가에게 확인하세요.",
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

# GroupChat 생성
group_chat = GroupChat(
    agents=[collector, cleaner, analyzer],
    messages=[],
    max_round=12  # 최대 대화 턴
)

# Manager 생성
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

# 대화 시작
collector.initiate_chat(
    manager,
    message="판매 데이터를 분석해야 합니다. 시작하죠!"
)
```

### GroupChatManager의 역할

**Orchestration의 Supervisor와의 차이:**

```python
# Supervisor (Orchestration)
"Agent A, 너는 데이터 수집해"
"Agent B, 너는 데이터 정제해"
→ 할 일까지 지정 (독재자형)

# GroupChatManager (AutoGen)
"지금 대화 흐름을 보니... 다음은 Agent A가 답변할 차례네"
→ 순서만 조율 (교통정리형)
```

**Manager는 LLM을 사용하여 다음 발언자를 선택:**
```
Turn 1: 데이터수집가: "sales.csv 수집 완료!"
Manager → "상황: 데이터 수집 완료. 다음은 정제가 필요하겠네"
         → 데이터정제가 선택

Turn 2: 데이터정제가: "음... 음수 값이 많은데요?"
Manager → "상황: 데이터 품질 문제 발생. 수집가에게 물어봐야겠네"
         → 데이터수집가 선택

Turn 3: 데이터수집가: "아, 환불 데이터라 음수예요"
Manager → "상황: 설명 들었으니 정제 계속하면 되겠네"
         → 데이터정제가 선택
```

### speaker_selection_method 옵션

```python
# 1. 자동 선택 (기본값) - Manager가 LLM으로 판단
group_chat = GroupChat(
    agents=[collector, cleaner, analyzer],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"
)

# 2. 순서대로 (round-robin)
speaker_selection_method="round_robin"
# 효과: collector → cleaner → analyzer → collector → ...

# 3. 랜덤
speaker_selection_method="random"

# 4. 수동 선택 (사람이 직접)
speaker_selection_method="manual"

# 5. 커스텀 함수
def custom_speaker_selection(last_speaker, groupchat):
    """
    last_speaker: 직전 발언자
    groupchat: GroupChat 객체
    return: 다음 발언자 Agent
    """
    last_message = groupchat.messages[-1]["content"]
    
    # "에러" 키워드가 있으면 수집가에게
    if "에러" in last_message or "이상" in last_message:
        return collector
    
    # "분석" 키워드가 있으면 분석가에게
    if "분석" in last_message:
        return analyzer
    
    # 기본: 정제가
    return cleaner

speaker_selection_method=custom_speaker_selection
```

### 실전 예제: 협업 데이터 분석

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Agent 1: 데이터 엔지니어
data_engineer = ConversableAgent(
    name="데이터엔지니어",
    system_message="""당신은 데이터 엔지니어입니다.
데이터 수집, 저장, 파이프라인 구축을 담당합니다.
데이터 품질 문제가 있으면 설명하세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=5
)

# Agent 2: 데이터 과학자
data_scientist = ConversableAgent(
    name="데이터과학자",
    system_message="""당신은 데이터 과학자입니다.
통계 분석, 머신러닝, 인사이트 도출을 담당합니다.
데이터가 이상하면 엔지니어에게 확인하세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=5
)

# Agent 3: 비즈니스 분석가
business_analyst = ConversableAgent(
    name="비즈니스분석가",
    system_message="""당신은 비즈니스 분석가입니다.
비즈니스 관점에서 데이터를 해석하고 의사결정을 지원합니다.
기술적 세부사항은 과학자에게 물어보세요.""",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_consecutive_auto_reply=5
)

# GroupChat 구성
group_chat = GroupChat(
    agents=[data_engineer, data_scientist, business_analyst],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "gpt-4", "api_key": "your-key"}
)

# 협업 시작
data_engineer.initiate_chat(
    manager,
    message="""
    고객 이탈률 분석 프로젝트를 시작합니다.
    - 데이터: customer_data.csv (10만 행)
    - 목표: 이탈 가능성 높은 고객 식별
    - 기한: 3일
    각자 역할에 맞게 의견 주세요.
    """
)
```

### 실제 대화 흐름 예시

```
Turn 1: 데이터엔지니어 (시작)
"customer_data.csv를 확인했습니다. 
- 결측치: 5%
- 중복: 200개
정제 후 분석 시작하겠습니다."

Turn 2: 데이터과학자 (Manager 선택)
"좋습니다. 정제 후 다음을 확인해주세요:
1. 이탈 고객 비율
2. 주요 feature들의 분포
이상치가 많으면 말씀해주세요."

Turn 3: 데이터엔지니어 (Manager 선택)
"정제 완료. 이탈률 23%입니다.
그런데 '최근_구매일' 컬럼에 미래 날짜가 있습니다."

Turn 4: 데이터과학자 (Manager 선택)
"엔지니어님, 미래 날짜는 시스템 버그로 보입니다.
제거하고 진행하시죠."

Turn 5: 비즈니스분석가 (Manager 선택)
"23% 이탈률이면 심각합니다.
과학자님, 어떤 고객군이 주로 이탈하나요?"

Turn 6: 데이터과학자 (Manager 선택)
"분석 중입니다. 주요 패턴:
- 최근 3개월 미구매 고객: 45% 이탈
- 고객센터 불만 2회 이상: 67% 이탈
비즈니스적으로 어떻게 해석하시나요?"

Turn 7: 비즈니스분석가 (Manager 선택)
"고객센터 개선이 시급하네요.
과학자님, 예측 모델 만들어주시면
마케팅팀에 타겟 리스트 전달하겠습니다."
```

### GroupChat 고급 설정

```python
# 발언 횟수 제한
group_chat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=20,
    speaker_selection_method="auto",
    allow_repeat_speaker=False  # 연속 발언 방지
)

# 특정 순서 강제
def strict_order_selection(last_speaker, groupchat):
    """순서 강제: agent1 → agent2 → agent3 → agent1"""
    order = [agent1, agent2, agent3]
    if last_speaker not in order:
        return order[0]
    idx = order.index(last_speaker)
    return order[(idx + 1) % len(order)]

group_chat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=12,
    speaker_selection_method=strict_order_selection
)

# 종료 조건 추가
def check_completion(message):
    content = message.get("content", "")
    # 모든 agent가 동의하면 종료
    return "모두 동의합니다" in content or "작업 완료" in content

# Agent에 종료 조건 설정
agent1 = ConversableAgent(
    name="Agent1",
    is_termination_msg=check_completion,
    ...
)
```

---

## 핵심 정리

### Step 2에서 배운 것

**2A: Reply 메커니즘**
- `max_consecutive_auto_reply`: 자동 응답 횟수 제한
- `is_termination_msg`: 종료 조건 감지
- 적절한 값: 5~10 (작업 복잡도에 따라)

**2B: Two-Agent 패턴**
- Coder (LLM) + Executor (실행)
- `system_message`로 행동 제어
- 코딩 어시스턴트 구현

**2C: Sequential Workflow**
- 단계별 agent pair 구성
- 파일 시스템으로 데이터 전달
- 명확한 워크플로우 처리

**2D: GroupChat**
- 여러 agent 자유 협업
- Manager가 발언 순서 조율
- 동적이고 유연한 문제 해결

### 패턴 선택 가이드

```python
# 간단한 작업 (계산, 간단한 코드)
→ Two-Agent 패턴 (Coder + Executor)

# 명확한 단계가 있는 작업 (ETL, 파이프라인)
→ Sequential Workflow

# 복잡하고 예측 불가능한 작업 (탐색적 분석)
→ GroupChat

# 정형화된 프로덕션 워크플로우
→ Orchestration (AutoGen 대신)
```

### 실전 체크리스트

**Agent 설계 시:**
- ✅ system_message 명확하게 작성
- ✅ 종료 조건 설정 (`is_termination_msg`)
- ✅ max_consecutive_auto_reply 적절하게 설정
- ✅ 작업에 맞는 패턴 선택

**디버깅 시:**
- ✅ 대화 로그 확인
- ✅ 각 agent의 응답 횟수 체크
- ✅ 종료 조건이 제대로 작동하는지 확인
- ✅ system_message가 오해의 소지 없는지 검토

---

## 다음 단계

Step 2를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ Reply 메커니즘 이해 및 제어
✅ Two-Agent 코딩 어시스턴트 구현
✅ Sequential Workflow 설계
✅ GroupChat으로 복잡한 협업 구현

**다음 학습 옵션**:

**Step 3A - 실전 프로젝트**
- 웹 스크래핑 + 분석 자동화
- 보고서 생성 파이프라인
- 멀티모달 데이터 처리

**Step 3B - 고급 기능**
- Function Calling
- RAG (Retrieval-Augmented Generation)
- Tool 통합

**Step 3C - 프로덕션 배포**
- 에러 처리
- 로깅 및 모니터링
- 비용 최적화
