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
### 1. Perception이란?

Perception은 Agent가 외부 환경으로부터 입력을 받아 **이해 가능한 형태로 변환하는 과정**임. 인간으로 치면 "보고 듣고 읽는" 감각 기관에 해당하며, 이후 Memory·Planning 등 모든 단계의 출발점이 됨.

아무리 뛰어난 추론 능력을 갖춘 Agent라도 입력을 제대로 인식하지 못하면 올바른 판단을 내릴 수 없음. 즉, **Perception의 품질이 Agent 전체 성능의 상한선을 결정**한다고 볼 수 있음.

Perception 단계에서 Agent가 수행하는 핵심 역할은 다음과 같음.

- **입력 수신** : 사용자 요청, 파일, API 응답 등 다양한 형태의 외부 입력을 받아들임.
- **형태 변환** : 수신한 입력을 LLM이 처리할 수 있는 형태로 변환함.
- **맥락 추출** : 입력에서 목표 달성에 필요한 핵심 정보를 추출함.

---
### 2. Input Types

Agent가 처리할 수 있는 입력 유형은 크게 **Text, Image, Audio** 세 가지로 분류됨. 각 유형은 처리 방식과 활용 목적이 상이하므로, Agent 설계 시 어떤 입력을 다룰지 명확히 정의하는 것이 중요함.

#### 2-1. Text

가장 기본적이고 널리 사용되는 입력 유형으로, 구조화 여부에 따라 두 가지로 구분됨.

- **Structured** : 명확한 형식을 가진 텍스트. JSON, XML, Table, CSV 등이 해당하며, 형식이 정해져 있어 파싱이 용이함.
- **Unstructured** : 형식이 정해지지 않은 텍스트. 자연어, 코드 등이 해당하며, 의미 파악을 위한 별도의 처리가 필요함.

#### 2-2. Image

시각적 정보를 담은 입력 유형으로, 표현 방식에 따라 두 가지로 구분됨.

- **Raster** : 픽셀 기반 이미지. PNG, JPEG, 스크린샷 등이 해당하며, 해상도에 따라 품질이 결정됨.
- **Vector** : 수학적 좌표 기반 이미지. SVG, 다이어그램 등이 해당하며, 해상도에 독립적으로 품질이 유지됨.

#### 2-3. Audio

소리 형태의 입력 유형으로, 음성 포함 여부에 따라 두 가지로 구분됨.

- **Speech** : 사람의 음성 입력. STT(Speech-to-Text)로 텍스트 변환 후 처리하거나, 화자 식별(Speaker ID)에 활용됨.
- **Non-speech** : 음성 외의 소리 입력. 환경음 분류(Sound Classification), 노이즈 제거(Noise Filtering) 등에 활용됨.

---
