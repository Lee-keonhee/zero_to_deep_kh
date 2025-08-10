---
layout: post
title:  "알고리즘 패러다임 - Greedy Algorithm"
summary: "Greedy Algorithm의 소개 및 예시"
author: keonhee
date: '2025-08-08 18:20:00 +0900'
category: 코테
#thumbnail: /assets/img/posts/propagation1.png
keywords: 알고리즘
permalink: /blog/algorithm paradigm/
usemathjax: true
---

### Greedy Algorithm

당장 눈앞에 보이는 최적의 선택

장점: 간단하고 빠르다.
단점: 최적의 답을 보장하지 않음

최적의 답이 필요없을때 사용 or 최적의 답을 구해주는 상황에서 사용

1.최적 부분 구조 - 부분 문제들의 최적의 답을 이용해서 기존 문제의 최적의 답을 구할 수 있는 것
2.탐욕적 선택 속성 - 각 단계에서의 탐욕스런 선택이 최종 답을 구하기 위한 최적의 선택

이 두개 만족하면 greedy로 최적 결과 구할 수 있음

