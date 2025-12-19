---
layout: post
title: MCP(Model Context Protocol)의 개요
summary: MCP(Model Context Protocol)의 개요
author: keonhee
date: 2025-12-12 10:00:00 +0900
category: MCP
keywords: MCP
permalink: /blog/MCP_1/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
# 01. MCP 개념 이해하기

## MCP란 무엇인가?

**MCP (Model Context Protocol)** = AI가 외부 도구/데이터를 사용할 수 있게 해주는 **표준 프로토콜**

### 왜 필요할까?

#### 기존 방식의 문제점

LangGraph로 RAG를 만들 때를 떠올려보세요:

```python
# 날씨 API 사용
weather = requests.get("https://weather-api.com/...")

# 데이터베이스 조회
db_result = sqlite3.connect("data.db").execute("SELECT...")

# 파일 읽기
with open("document.txt") as f:
    content = f.read()
```

**문제점:**

- 각 도구마다 다른 방식으로 연결
- 새 도구 추가할 때마다 코드 수정 필요
- 다른 프로젝트에서 재사용하기 어려움
- 다른 언어로 작성된 프로젝트에서는 사용 불가

#### MCP의 해결책

모든 도구를 **같은 방식**으로 사용할 수 있게 만듭니다!

```
Claude (AI) ←→ MCP Protocol ←→ 도구들 (날씨, DB, 파일...)
```

마치 날씨 API처럼:

- 내부가 어떻게 만들어졌든 상관없이
- 표준화된 방식으로 호출
- 어디서든 사용 가능

---

## MCP의 핵심 구성 요소

### 1. MCP Server (서버)

- **역할**: 도구/데이터를 제공하는 쪽
- **예시**: 문서 검색 서버, 날씨 조회 서버, DB 접근 서버
- **여러분이 만들 것**: Python으로 작성된 MCP 서버

### 2. MCP Client (클라이언트)

- **역할**: 서버의 기능을 사용하는 쪽
- **예시**: Claude, 여러분의 애플리케이션
- **특징**: 서버가 어떤 언어로 만들어졌는지 몰라도 됨

### 3. Protocol (프로토콜)

- **역할**: 서버와 클라이언트가 소통하는 규칙
- **특징**: JSON 기반의 표준화된 메시지 형식

---

## 실제 비유로 이해하기

### 음식점으로 비유하면:

|구성요소|음식점|MCP|
|---|---|---|
|**서버**|주방 (요리 만듦)|MCP Server (기능 제공)|
|**클라이언트**|손님 (음식 주문)|Claude/앱 (기능 사용)|
|**프로토콜**|메뉴판 (주문 방식)|MCP Protocol (통신 규칙)|

- 손님은 주방이 어떻게 요리하는지 몰라도 됨
- 메뉴판만 보고 주문하면 됨
- 주방이 바뀌어도 메뉴판 형식은 동일

---

## MCP vs 기존 방식 비교

### 기존 방식 (LangGraph + 직접 구현)

```python
# 각 도구마다 다른 코드
def search_docs(query):
    # 문서 검색 로직
    return results

def get_weather(city):
    # 날씨 API 호출
    return weather

# LangGraph 노드에 직접 포함
graph = StateGraph(...)
graph.add_node("search", search_docs)
graph.add_node("weather", get_weather)
```

**문제점:**

- ❌ 다른 프로젝트에서 재사용 어려움
- ❌ 새 도구마다 그래프 수정 필요
- ❌ Python 외 언어에서 사용 불가

### MCP 방식

```python
# 1. MCP 서버 만들기 (한 번만)
# 문서 검색 MCP 서버
# 날씨 조회 MCP 서버

# 2. Claude가 필요할 때 자동으로 호출
# 별도 코드 작성 불필요!
```

**장점:**

- ✅ 한 번 만들면 어디서든 재사용
- ✅ 새 도구 추가해도 기존 코드 수정 불필요
- ✅ 언어 무관하게 사용 가능
- ✅ Claude가 스스로 판단해서 도구 선택

---

## 학습 로드맵

이제부터 이런 순서로 배워볼 거예요:

```
✓ 01. 개념 이해 ← 지금 여기!
→ 02. 첫 번째 MCP 서버 만들기 (간단한 계산기)
→ 03. 도구 추가하기 (여러 기능 제공)
→ 04. Claude와 연결하기 or 로컬 모델과 연결하기
→ 05. 실전 프로젝트 (RAG를 MCP로 전환)
```

---

## 체크포인트: 이해했는지 확인해보세요

다음 질문에 답할 수 있으면 다음 단계로 넘어갈 준비가 된 거예요:

1. **MCP가 해결하는 문제는 무엇인가요?**
    
    - 힌트: API마다 다른 방식으로 연결해야 하는 문제
    - 정답: 언어와 관계 없이 모든 도구를 같은 방식으로 불러서 사용할 수 있다.
2. **MCP Server의 역할은 무엇인가요?**
    
    - 정답: 도구/데이터를 제공하는 쪽
3. **MCP를 사용하면 어떤 장점이 있나요?**
    
    - 정답: 재사용성, 언어 독립성...

---

## 다음 단계

이제 실제로 MCP 서버를 만들 준비를 해볼까요?

**다음: [MCP-서버 만들기](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_2/#/)**

Windows + Python 3.12 + PyCharm 환경에 MCP를 설치하고 설정하는 방법을 배워봅시다!