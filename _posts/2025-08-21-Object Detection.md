---
layout: post
title:  "객체 탐지(Object Detection)"
summary: "객체 탐지의 기본과 1-stage, 2-stage object detection"
author: keonhee
date: '2025-08-21 12:20:00 +0900'
category: Deeplearning, Object detection
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/Object_detection/
usemathjax: true
---
<hr class="thick-hr">

### Object Detection

<hr class="thick-hr">

이전 시간에는, CNN과 CNN 기반의 여러 모델에 대해서 공부하고 이를 통해, 이미지의 분류를 하는 법을 알아보았습니다. 이번 시간에는, backbone 모델(데이터에서 유용한 특징을 추출하는 역할을 담당하는 네트워크)에서 추출된 Feature map을 이용하여 이미지의 객체를 탐색하는 Object Detection에 대해서 알아보겠습니다.

#### 객체탐지(Object Detection)란
객체 탐지는 컴퓨터 비전 기술의 세부 분야중 하나로써 주어진 이미지내 사용자가 관심 있는 객체를 탐지하는 기술입니다. 이전의 분류 모델들과 비교를 해보겠습니다.
인공지능 모델이 그림 좌측의 강아지 사진을 강아지라고 판별한다면 해당 모델은 이미지 분류 모델 입니다. 하지만 우측 사진 처럼 물체가 있는 위치를 탐지함과 동시에 해당 물체가 강아지라고 분류 한다면 해당 모델은 객체 탐지 모델입니다.
<img src="{{"/assets/img/posts/classification_objectdetection.jpg" | relative_url }}" width="600" height="400" alt="객체 탐지와 분류" style="display: block; margin: 0 auto;">

이처럼 Object Detection은 물체의 위치를 Bounding box를 통해 **위치를 파악**하고 그 물체가 어떤 물체인지 **분류**를 할 수 있습니다. 또한, Object Detection은 Object의 수에 따라서 하나의 물체를 찾는 **Single-object Detection**과 여러 물체를 찾는 **Multi-object Detection**이 있습니다.
<img src="{{"/assets/img/posts/object_detection.png" | relative_url }}" width="600" height="400" alt="다중 객체 탐지" style="display: block; margin: 0 auto;">

객체 탐지는 수행 방식에 따라 크게 두가지로 나눌 수 있습니다.
1. 1-stage Object Detection
1-stage Object Detection의 경우, Region proposal과 Classification을 동시에 수행합니다. 동시에 수행함에 따라, 속도가 빠르다는 장점이 있지만 정확도가 좋지 않다는 특징이 있습니다.
<br>
2. 2-stage Object Detection
2-stage Object Detection의 경우, Region proposal을 수행한 이후 Classification을 수행합니다. 순차적으로 수행함에 따라, 속도가 느리지만 정확도가 뛰어나다는 장점이 있습니다.

| 특징 | 1-스테이지 객체 탐지 (One-stage Object Detection) | 2-스테이지 객체 탐지 (Two-stage Object Detection) |
|---|---|---|
| **핵심 동작** | Region proposal (영역 제안)과 Classification (분류)을 동시에 수행 | Region proposal (영역 제안)을 수행한 이후 Classification (분류)을 순차적으로 수행 |
| **장점** | 속도가 빠르며, 실시간 탐지에 유리 | 정확도가 뛰어나며, 특히 작은 객체 탐지에 강점 |
| **단점** | 초기에는 정확도가 2-스테이지에 비해 다소 낮았음 (최근에는 많이 개선) | 속도가 느리며, 실시간 탐지에 적용하기 어려울 수 있음 |
| **대표 모델** | - YOLO (You Only Look Once) 계열<br/>- SSD (Single Shot MultiBox Detector) | - R-CNN (Region-based CNN)<br/>- Fast R-CNN<br/>- Faster R-CNN<br/>- Mask R-CNN |

이 중 먼저 연구가 진행이 되었던 2-stage object detection에 대해서 알아보겠습니다.

<hr class="thin-hr">

### 2-stage object detecion

2-stage object detection 모델은 객체 탐지를 **두 단계의 과정**으로 나누어 수행하는 방식입니다.

1. 1단계 - 객체 후보 영역 제안(Region Proposal) : 이미지에서 **객체가 있을 만한 위치**를 효율적으로 찾습니다.
2. 2단계 - 후보 영역 분류 및 바운딩 박스 정제 (Classification and Refinement) : 후보 영역(RoI)에 어떤 객체가 있는지 분류하고 박스를 더욱 정확하게 확정합니다.

#### 2-stage 객체 탐지의 종류

| 모델            | 단계 | 내용                                                      |
|-----------------|------|---------------------------------------------------------|
| **R-CNN**       | 1단계 | 선택적 탐색(Selective Search) 알고리즘으로 후보 영역 추출, 각 후보를 CNN에 통과 |
|                 | 2단계 | SVM으로 분류 + Regressor로 바운딩 박스 조정                         |
|                 | 한계  | 각 후보마다 CNN 수행 → 속도 매우 느림                                |
| **Fast R-CNN**  | 1단계 | 전체 이미지를 CNN 1번 통과 → 후보 영역 추출, ROI Pooling으로 고정 크기 특징맵   |
|                 | 2단계 | 분류 + 바운딩 박스 회귀 동시 수행                                    |
|                 | 한계  | 후보 영역 추출이 여전히 Selective Search라 느림                      |
| **Faster R-CNN**| 1단계 | RPN(Region Proposal Network)으로 후보 직접 제안                 |
|                 | 2단계 | ROI Pooling 후 분류 + 박스 refine                            |
|                 | 한계  | 작은 객체 취약, 원스테이지보다 느림                                    |
| **Mask R-CNN**  | 1단계 | Faster R-CNN과 동일, RPN 사용                                |
|                 | 2단계 | ROI Align 적용 → 분류 + 박스 회귀 + 마스크 예측                      |
|                 | 한계  | 인스턴스 분할 추가 → 계산량 증가, 속도 더 느림                            |


### Faster R-CNN

2단계 객체탐지 모델 중 가장 보편적으로 사용되는 Faster R-CNN에 대해서 조금 더 알아보도록 하겠습니다.


#### 1단계 : 객체 후보 영역 제안(Region Proposal)

이 단계에서는 이미지에서 **객체가 있을 만한 위치**를 효율적으로 찾습니다.

1. 백본 모델로부터 특징 맵(Feature Map) 받기
- 입력 이미지는 CNN 기반의 네트워크를 통과하여 **특징 맵(Feature Map)**을 생성합니다.
<br>
2. 영역 제안 네트워크(Region Proposal Network, RPN)의 역할:
- 생성된 특징 맵은 RPN이라는 특별한 서브 네트워크의 입력으로 들어갑니다.
- RPN은 이 특징 맵 위에서 **앵커 박스(Anchor Box)**를 기반으로 작동합니다. 각 특징 맵 위치(픽셀)에 대해 미리 정의된 다양한 크기와 비율의 앵커 박스들이 투영됩니다.
<br>
※ 앵커박스의 크기를 다양하게 하는 이유?
학습이 진행되면서 앵커박스의 크기와 비율이 target에 맞게 조정이 됩니다. 하지만 앵커박스의 크기와 비율을 큰 값으로 조정을 해야할 경우, 큰 폭의 offset과 scale 변화를 학습해야 하기때문에 학습이 어려워 집니다. 뿐만 아니라, 이러한 과정은 학습이 비효율, 불안정해져 학습의 속도가 느려진다는 단점이 있습니다.
<br>

3. RPN의 예측 (Learning to moeve Anchor Boxes)
- RPN은 각 앵커 박스에 대해 두 가지를 예측합니다.
  - **객체성 점수(Objectness Score)** : 앵커 박스 내에 객체가 있는가 없는가(이 박스가 객체일 가능성이 얼마나 높은가). 이는 0과 1 사이의 확률 값으로 나타납니다.
  - **바운딩 박스 변환값(Bounding Box Regressions)**: 만약 객체가 있다면, 이 앵커박스의 위치와 크기를 실체 객체에 더 맞추기 위해 **얼마나 이동(offset)시키고, 크기를 조절(scale)해야하는지** $$(t_x, t_y, t_w, t_h)$$ 를 예측합니다.
- 신경망은 바로 이 $$(t_x, t_y, t_w, t_h)$$ 값을 예측하고 학습합니다. 이 과정에서 IoU를 사용하여 예측된 박스가 정답 박스에 얼마나 가까운지 평가하고, 손실 함수를 통해 모델의 파라미터를 업데이트합니다.
<br>

※ IOU란 무엇인가?
IOU란 객체탐지에 사용되는 평가지표로써, Intersection over Union의 약자입니다. **예측된 바운딩 박스**와 **실제 바운딩박스**가 얼마나 겹치는지를 수치화한 값으로, 객체의 위치와 크기를 얼마나 정확하게 예측했는지를 평가합니다.
IOU는 $$IOU = \frac{두 박스의 교집합 면적}{두 박스의 합집합 면적}$$ 으로 계산됩니다. 0-1 사이의 값을 가지며, 커질수록 모델의 예측이 정확하다는 것을 의미합니다.

<br>

4. 후보 영역(Rigion Proposal)추출
- RPN이 예측한 수많은 앵커 박스들 중에서, 객체성 점수가 높은 박스들을 우선 선택합니다.
- 이후 NMS(Non-Maximum Suppression)과 같은 후처리 과정을 거쳐, 중복되는 박스들을 제거하고, 가장 가능성이 높은 **객체 후보 영역(Region Proposals)**를 수백 수천 개를 선택합니다.

※ NMS란 무엇인가?
Non-Maximum Suppression의 약자로,
111
#### 2단계: 후보 영역 분류 및 바운딩 박스 정제 (Classification and Refinement)
이 단계의 목표는 **후보 영역에 어떤 객체가 있는지 분류하고 박스를 더욱 정확하게 확정하는 것'**입니다.
1. ROI 풀링 (Region of Interest Pooling):
- 1단계에서 제안된 각 객체 후보 영역(Region Proposal)은 특징 맵 위에서 해당하는 영역(Region of Interest, ROI)을 나타냅니다.
- 이 ROI들은 크기가 제각각일 수 있는데, 이들을 모두 고정된 크기(예: 7x7)의 특징 벡터로 만들어주는 **ROI 풀링(ROI Pooling)**이라는 과정이 필요합니다. 이는 다음 단계의 분류 및 회귀 네트워크에 입력하기 위함입니다.
2. 최종 분류기 (Classifier Head) 및 회귀기 (Regressor Head):
- ROI 풀링을 거쳐 고정된 크기의 특징 벡터가 나오면, 이 벡터는 두 갈래의 서브 네트워크로 들어갑니다:
  - (a) 최종 분류기 (Classifier Head): 이 후보 영역 안에 있는 객체가 정확히 '어떤 클래스(예: 사람, 자동차, 자전거 등)'에 해당하는지를 분류합니다. 배경일 경우도 포함하여 분류합니다.
  - (b) 최종 바운딩 박스 회귀기 (Regressor Head): 1단계 RPN에서 제안된 박스를 보다 정밀하게 조정하여 최종 바운딩 박스를 결정합니다. 이때도 역시 t_x, t_y, t_w, t_h와 유사한 개념의 변환 값을 예측합니다. 이 변환 값은 1단계에서 예측한 t 값보다 더 미세하게 박스를 조정합니다.
3. 최종 객체 탐지 결과:
- 이 2단계의 과정을 통해 이미지 내의 각 객체에 대해 정확한 클래스 라벨과 정제된 바운딩 박스 좌표가 도출됩니다.

