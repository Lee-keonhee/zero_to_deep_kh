---
layout: post
title: Agent 시스템 개요
summary: agent 시스템의 주요 요소
author: keonhee
date: 2025-03-30 12:00:00 +0900
category: Agent system
keywords: Agent
permalink: /blog/agent_system/
usemathjax: false
thumbnail: /assets/img/posts/agent_system_tree.png
imageNameKey: Agent System
---
![](assets/img/posts/agent_system_tree.png)
# Agent 시스템

### 1. Agent는 왜 필요할까?
#### LLM의 한계
- LLM은 질문에 "답변"만 할 뿐, 스스로 "행동"하지 못함
- 학습된 데이터 기반이라 실시간 정보에 접근 불가
- 한 번의 응답으로 끝나기 때문에 복잡한 다단계 작업 수행 불가
- 계산, 파일 처리 등 실제 작업을 직접 실행할 수 없음

### Agent 시스템의 정의
Agent 시스템이란, 사용자의 목표를 전달받아, 스스로 계획을 세우고 도구를 사용해 그 목표를 완수하는 **자율적 시스템**

1. LLM vs Agent 

| 구분  | LLM(Model)               | Agent(System)          |
| --- | ------------------------ | ---------------------- |
| 역할  | 텍스트 생성 및 추론( 지식의 창고)     | 목표 달성을 위한 실행( 행동의 주체 ) |
| 특징  | 수동적 (질문에 대답)             | 능동적(필요한 단계를 스스로 결정)    |
| 상태  | Stateless(이전 대화를 기억 못 함) | Stateful(기억과 맥락을 유지)   |

2. Agent의 3대 핵심요소
	에이전트 시스템이 성립되기 위해서는 아래 3가지 요소가 유기적으로 연결되어야 함
	- **자율성(Autonomy)** : 인간이 일일이 지시하지 않아도, 최종 목표만 주어지면 세부 실행 단계는 스스로 판단함.
	- **지속성(Persistence)** : 한 번의 응답으로 끝나는 것이 아니라, 목표가 달성될 때까지 프로세스를 유지하며 환경과 상호작용
	- **도구 활용(Tool Use)** : 자신의 내부 지식만 아니라 계산기, 검색 엔진, API 등 외부 도구를 적재적소에 활용

3. 에이전트의 작동 루프
	에이전트는 보통 다음과 같은 순환 구조를 가짐. 이를 ReAct(Reason+Act) 패턴이라고도 함.
		1. Perceive(인식) : 사용자의 요청ㅇ이나 환경 변화를 감지
		2. Think(추론) : 지금 상황에서 무엇을 해야 목표에 가까워질까 고민
		3. Plan(계획) : 실행할 구체적인 단계나 도구 선택
		4. Act(실행) : 실제 행동(코드 실행, API 호출 등)
		5. Observe(관찰) : 실행 결과를 보고 성공 여부를 판단한 뒤 다시 2번으로 돌아감
	