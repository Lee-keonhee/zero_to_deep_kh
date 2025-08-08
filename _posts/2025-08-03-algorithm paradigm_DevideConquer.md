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


예시 2) 합병정렬

예시 2-1) 정렬된 리스트를 받아서 새로운 정렬된 리스트 생성하는 merge 함수 구현하기

```python
def merge(list1, list2):
    merged_list = []
    x = 0
    y = 0
    # 하나의 리스트만 소진되어도 while문 종료/ while문 반복하는 동안 merged_list에 각 리스트의 원소 비교하며 합병 
    while x <len(list1) and y<len(list2):
        if list1[x] > list2[y]:
            merged_list.append(list2[y])
            y += 1
        else:
            merged_list.append(list1[x])
            x += 1
    # 소진되지 않은 나머지 리스트를 합병
    if x == len(list1):
        merged_list += list2[y:]
    else:
        merged_list += list1[x:]
        
    return merged_list

# 테스트 코드
print(merge([1],[]))                        # [1]
print(merge([],[1]))                        # [1]
print(merge([2],[1]))                       # [1,2]
print(merge([1, 2, 3, 4],[5, 6, 7, 8]))     # [1,2,3,4,5,6,7,8]
print(merge([5, 6, 7, 8],[1, 2, 3, 4]))     # [1,2,3,4,5,6,7,8]
print(merge([4, 7, 8, 9],[1, 3, 6, 10]))    # [1,3,4,6,7,8,9,10]
```

예시 2-2) Divide and Conquer 방식으로 합병 정렬하기
```python
def merge_sort(my_list):
    if len(my_list)<2:
        return my_list
    return merge(merge_sort(my_list[:len(my_list)//2]), merge_sort(my_list[len(my_list)//2:]))

```

예시 3) 퀵정렬
특정 값을 pivot으로 설정하고 왼쪽에는 pivot보다 작은값, 오른쪽엔 큰값을 집어넣음을 반복하면서 정렬시킴
```python
def swap_elements(my_list, index1, index2):
    my_list[index1], my_list[index2] = my_list[index2], my_list[index1]
    return my_list

def partition(my_list, start, end):
    p = end
    i = start
    b = start
    while i < p:
        if my_list[i] <= my_list[p]:
            my_list = swap_elements(my_list, i, b)
            b += 1
        i += 1
    my_list = swap_elements(my_list, b,p)
    p = b
    return p

def quicksort(my_list, start=0, end=None):
    if end == None:
        end = len(my_list) -1
    # 여기에 코드를 작성하세요
    if end - start < 1:
        return

    # my_list를 두 부분으로 나누어주고,
    # partition 이후 pivot의 인덱스를 리턴받는다
    pivot = partition(my_list, start, end)

    # pivot의 왼쪽 부분 정렬
    quicksort(my_list, start, pivot - 1)

    # pivot의 오른쪽 부분 정렬
    quicksort(my_list, pivot + 1, end)
```