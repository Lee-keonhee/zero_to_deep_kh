---
layout: post
title:  "알고리즘 패러다임 - Dynamic Programming"
summary: "Dynamic Programming의 소개 및 예시"
author: keonhee
date: '2025-08-08 12:20:00 +0900'
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
중복 + 최적 구분구조 - Dynamic Programming은 두 가지 방법이 있음 - Memmoization, Tabulation

### Memmoization
종복된 계산은 한번만 계산 후 메모함. Top-down 방식

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

### Tabulation
Down-Top 방식

```python
def fib_tab(n):
    table= [0, 1, 1]
    for i in range(3, n+1):
        table.append(table[n-1] + table[n-2])
    return table[n]
```

하지만 피보나치 수를 계산하려면, 이전 두개의 숫자만 알면됨. 

```python
def fib_opt(n):
    prvious = 1
    current = 1
    for _ in range(2, n):
        previous, current = current, previous+current
    return current

```
해당 방식을 통해, 공간 복잡도가 O(n)에서 O(1)로 감소함.

### Tabulation과 Memmoization
Tabulation의 경우, 아래부터 하나하나 다 계산하는 방식. 모든 계산을 다 해야함.
Memmoization은 필요한 계산만 하게됨.

최대 이득을 계산하는 문제
***Memoization***
price_list: 개수별 가격이 정리되어 있는 리스트
count: 판매할 물품 개수
cache: 개수별 최대 수익이 저장되어 있는 사전

```python
def max_profit_memo(price_list, count, cache):
    # count, 즉 팔수 있는 개수가 2개 이하일, 경우,
    if count < 2:
        cache[count] = price_list[count]
        return cache[count]
    if count in cache:
        return cache[count]
    if count <= len(price_list)-1:
        profit = price_list[count]
    else:
        profit=0
    for i in range(1, count//2 + 1):
        profit = max(profit, max_profit_memo(price_list, i, cache) + max_profit_memo(price_list, count-i, cache))
    cache[count] = profit
    return cache[count]
    

def max_profit(price_list, count):
    max_profit_cache = {}
    return max_profit_memo(price_list, count, max_profit_cache)

# 테스트 코드
print(max_profit([0, 100, 400, 800, 900, 1000], 5))                                     # 1200
print(max_profit([0, 100, 400, 800, 900, 1000], 10))                                    # 2500
print(max_profit([0, 100, 400, 800, 900, 1000, 1400, 1600, 2100, 2200], 9))             # 2400

```

같은 문제를 Tabulation으로 풀어보기

```python
def max_profit(price_list, count):
    # max profit table 설정
    profit_table = price_list[:2]
    # 2 이하일 때는 바로 출력
    if count < 2:
        return profit_table[count]
    # profit 비교하기
    for i in range(2,count+1):
        if i <= len(price_list)-1:
            profit = price_list[i]
        else:
            profit = 0
        for j in range(1, i//2 +1):
            profit = max(profit, profit_table[j]+profit_table[i-j])
        profit_table.append(profit)
    return profit_table[count]


```
