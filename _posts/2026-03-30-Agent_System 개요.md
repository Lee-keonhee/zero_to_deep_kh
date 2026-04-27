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
4