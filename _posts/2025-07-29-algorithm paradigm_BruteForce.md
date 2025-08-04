---
layout: post
title:  "알고리즘 패러다임 - Brute Force"
summary: "Brute Foce의 소개 및 예시"
author: keonhee
date: '2025-07-29 16:20:00 +0900'
category: 코테
#thumbnail: /assets/img/posts/propagation1.png
keywords: 알고리즘
permalink: /blog/algorithm paradigm/
usemathjax: true
---

### Brute Force 알고리즘

모든 데이터를 하나씩 확인해 나가면서, 원하는 결과를 내는 데이터를 추출하는 알고리즘

예시 1) 왼쪽카드 뭉치, 오른쪽 카드 뭉치에서 카드를 한장씩 뽑아서 두 수의 곱이 가장 큰 결과값을 도출하세요.

```python
def max_product(left_cards, right_cards):
    mul_list = []
    for left in left_cards:
        for right in right_cards:
            mul_list.append(left*right)
    return max(mul_list)

print(max_product([1, 6, 5], [4, 2, 3]))
print(max_product([1, -9, 3, 4], [2, 8, 3, 1]))
print(max_product([-1, -7, 3], [-4, 3, 6]))
```


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