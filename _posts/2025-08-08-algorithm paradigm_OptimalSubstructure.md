---
layout: post
title:  "알고리즘 패러다임 - Dynamic Programming"
summary: "Dynamic Programming의 소개 및 예시"
author: keonhee
date: '2025-08-08 16:20:00 +0900'
category: 코테
#thumbnail: /assets/img/posts/propagation1.png
keywords: 알고리즘
permalink: /blog/algorithm paradigm/
usemathjax: true
---

### Optimal Substructure의 구조

최적 부분 구조가 있다는 것은 부분문제들의 최적의 답을 이용해서 기존 문제의 최적의 답을 구할 수 있다는 것을 의미
즉, 기존문제를 부분문제로 나눠서 풀수 잇음. 이때, 중복되는 부분문제가 있을 수 있음 - Dynamic Programming 활용

Dynamic Programming - 한번 계산한 결과를 재활용하는 방식


### 중복되는 부분 문제 vs 중복되지 않는 부분문제


### Memmoization
종복된 계산은 한번만 계산 후 메모함.

cache - 한번 계산한걸 저장하는 곳

cache를 이용한 피보나치 수열 계산과 기본 피보나치 수열 계산 차이
```python
# 기본 피보나치 수열
def fibo(n):
    if n < 3:
        return 1
    return fibo(n-1) + fibo(n-2)

```
기본 피보나치 수열에서 n이 10이라고 할때, fibo(8)과 fibo(9)를 각각 계산하는 데, 중복된 부분 즉 fibo(9) = fibo(8) + fibo(7)에서 fibo(8)부분을 다시 계산하여, 같은 부분을 다시 계산하여 계산량이 많아지게된다.

```python
def fib_memo(n, cache):
    # 여기에 코드를 작성하세요
    if n < 3:
        return 1
    if n in cache:
        return cache[n]
    else:
        cache[n] = fib_memo(n-1, cache) + fib_memo(n-2, cache)
        return cache[n]
    
def fib(n):
    # n번째 피보나치 수를 담는 사전
    fib_cache = {}
    
    return fib_memo(n, fib_cache)
```