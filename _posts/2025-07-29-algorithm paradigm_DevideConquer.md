---
layout: post
title:  "알고리즘 패러다임 - Devide and Conquer"
summary: "Devide and Conquer의 소개 및 예시"
author: keonhee
date: '2025-07-29 16:20:00 +0900'
category: 코테
#thumbnail: /assets/img/posts/propagation1.png
keywords: 알고리즘
permalink: /blog/algorithm paradigm/
usemathjax: true
---

### Devide and Conquer 알고리즘

해결해야할 문제를 나누어, 작은 문제를 해결하고 그 해결한 값을 합하여 큰 문제를 해결하는 방식(재귀와 비슷한 방식이므로, 재귀에 대한 설명을 이해하고 공부하도록 하자)

#### ***재귀함수***

재귀 함수는 함수 내부에서 자기 자신을 다시 호출하는 함수를 의미한다. 재귀함수는 반복문(for, while)이 없어도 같은 작업을 수행할 수 있으며, 무한반복에 빠지지 않도록 조심해야한다.
만약 무한 반복에 빠질경우, stack overflow가 발생할 수 있다. Python의 경우, 최대 재귀 깊이(maximum recursion depth)가 1000이다.

예시 1) n번째 삼각수( 정수 1부터 n까지의 합) 구하기

```python
def triangle_number(n):
    if n <=1:
        return n
    return triangle_number(n-1) + n
```
위와 같이 def 함수 내에 자기자신을 호출하여 반복문 작업을 수행하는 함수를 **재귀함수**라고 한다.


#### Devide and Conquer 알고리즘 예시

예시 1) Divide and Conquer를 이용해서 1부터 n까지 더하는 예시
```python
def consecutive_sum(start, end):
    if start == end:
        return start
    return consecutive_sum(start,(start+end)//2) + consecutive_sum((start+end)//2 + 1, end)
    
# 테스트 코드
print(consecutive_sum(1, 10))
print(consecutive_sum(1, 100))
print(consecutive_sum(1, 253))
print(consecutive_sum(1, 388))
```

해당 함수를 보면, consecutive_sum함수를 계속해서 나누어, 하나의 수만 남겨 각각의 값을 반환하여 하나씩 더하는 것을 확인할 수 있다. 이렇게 문제를 작은 문제로 나누고 이를 하나씩 해결하므로써 해당 문제를 해결하는 방법을 Devide and Conquer라고 한다.


예시 2) 가장 가까운 매장 찾기

```python
from math import sqrt

# 두 매장의 직선 거리를 계산해 주는 함수
def distance(store1, store2):
    return sqrt((store1[0] - store2[0]) ** 2 + (store1[1] - store2[1]) ** 2)

# 가장 가까운 두 매장을 찾아주는 함수
def closest_pair(coordinates):
     min_dist = distance(coordinates[0],coordinates[1])
    min_dist_store = []
    for i in test_coordinates:
        for j in test_coordinates:
            
            if i == j:
                pass
            else:
                dist = distance(i,j)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_store = [i,j] 
    return min_dist_store

# 테스트 코드
test_coordinates = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
print(closest_pair(test_coordinates))
```


예시 3) 런던 폭우 문제
다음 그림과 같이 폭우가 와서, 건물이 비에 잠길 정도입니다. 이때 얼만큼의 빗물이 담길 수 있는지 계산해 주는 함수를 작성해 보세요

![폭우](/assets/img/posts/londonRain1.png)

![폭우](/assets/img/posts/londonRain2.png)

```python
def trapping_rain(buildings):
    # 여기에 코드를 작성하세요
    total_rain = 0
    front_index = 0
    for i in range(1, len(buildings)-1):
        left_max = max(buildings[:i])
        right_max = max(buildings[i+1:])
        total_rain += max(0,min(left_max,right_max)-buildings[i])
    return total_rain
            
# 테스트
print(trapping_rain([3, 0, 0, 2, 0, 4]))
print(trapping_rain([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))

```