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


<hr class="thin-hr">

### 2-stage object detecion

2-stage object detection 모델은 객체 탐지를 **두 단계의 과정**으로 나누어 수행하는 방식입니다.

1. 1단계 - 객체 후보 영역 제안(Region Proposal) : 이미지에서 **객체가 있을 만한 위치**를 효율적으로 찾습니다.
2. 2단계 - 후보 영역 분류 및 바운딩 박스 정제 (Classification and Refinement) : 후보 영역(RoI)에 어떤 객체가 있는지 분류하고 박스를 더욱 정확하게 확정합니다.

#### 2-stage 객체 탐지의 종류
| 모델            | 단계 | 내용 |
|-----------------|------|------|
| **R-CNN**       | 1단계 | 선택적 탐색(Selective Search) 알고리즘으로 후보 영역 추출, 각 후보를 CNN에 통과 |
|                 | 2단계 | SVM으로 분류 + Regressor로 바운딩 박스 조정 |
|                 | 한계  | 각 후보마다 CNN 수행 → 속도 매우 느림 |
| **Fast R-CNN**  | 1단계 | 전체 이미지를 CNN 1번 통과 → 후보 영역 추출, ROI Pooling으로 고정 크기 특징맵 |
|                 | 2단계 | 분류 + 바운딩 박스 회귀 동시 수행 |
|                 | 한계  | 후보 영역 추출이 여전히 Selective Search라 느림 |
| **Faster R-CNN**| 1단계 | RPN(Region Proposal Network)으로 후보 직접 제안 |
|                 | 2단계 | ROI Pooling 후 분류 + 박스 refine |
|                 | 한계  | 작은 객체 취약, 원스테이지보다 느림 |
| **Mask R-CNN**  | 1단계 | Faster R-CNN과 동일, RPN 사용 |
|                 | 2단계 | ROI Align 적용 → 분류 + 박스 회귀 + 마스크 예측 |
|                 | 한계  | 인스턴스 분할 추가 → 계산량 증가, 속도 더 느림 |


### Faster R-CNN
여러 2스테이지 객체탐지 모델 중 가장 보편적으로 사용 되는 Faster R-CNN에 대해서 조금 더 알아보도록 하자.

#### 1단계 : 객체 후보 영역 제안(Region Proposal)

이 단계에서는 이미지에서 **객체가 있을 만한 위치**를 효율적으로 찾습니다.

1. 백본 모델로부터 특징 맵(Feature Map) 받기
- 입력 이미지는 CNN 기반의 네트워크를 통과하여 **특징 맵(Feature Map)**을 생성합니다.
<br>
2. 영역 제안 네트워크(Region Proposal Network, RPN)의 역할:
- 생성된 특징 맵은 RPN이라는 특별한 서브 네트워크의 입력으로 들어갑니다.
- RPN은 이 특징 맵 위에서 **앵커 박스(Anchor Box)**를 기반으로 작동합니다. 각 특징 맵 위치(픽셀)에 대해 미리 정의된 다양한 크기와 비율의 앵커 박스들이 투영됩니다.

[//]: # (- 왜 크기를 다양하게하지? 일정하게 해도 상관없는거 아닌가 결국 parameter로 크기 변환하는데?)
<br>

3. RPN의 예측 (Learning to moeve Anchor Boxes)
- RPN은 각 앵커 박스에 대해 두 가지를 예측합니다.
  - **객체성 점수(Objectness Score)** : 앵커 박스 내에 객체가 있는가 없는가(이 박스가 객체일 가능성이 얼마나 높은가). 이는 0과 1 사이의 확률 값으로 나타납니다.
  - **바운딩 박스 변환값(Bounding Box Regressions)**: 만약 객체가 있다면, 이 앵커박스의 위치와 크기를 실체 객체에 더 맞추기 위해 **얼마나 이동(offset)시키고, 크기를 조절(scale)해야하는지** $$(t_x, t_y, t_w, t_h)$$ 를 예측합니다.
- 신경망은 바로 이 $$(t_x, t_y, t_w, t_h)$$ 값을 예측하고 학습합니다. 이 과정에서 IoU를 사용하여 예측된 박스가 정답 박스에 얼마나 가까운지 평가하고, 손실 함수를 통해 모델의 파라미터를 업데이트합니다.

[//]: # (IOU에 대해서 알아보자)

<br>

4. 후보 영역(Rigion Proposal)추출
- RPN이 예측한 수많은 앵커 박스들 중에서, 객체성 점수가 높은 박스들을 우선 선택합니다.
- 이후 NMS(Non-Maximum Suppression)과 같은 후처리 과정을 거쳐, 중복되는 박스들을 제거하고, 가장 가능성이 높은 **객체 후보 영역(Region Proposals)**를 수백 수천 개를 선택합니다.

[//]: # (NMS에 대해서 좀 더 알아보자)


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



#### MLP의 한계점
전통적인 MLP(Multi Layer Perceptron)의 경우, 여러 층의 퍼셉트론을 통해 비선형 문제를 해결할 수 있었지만, 이미지나 영상을 처리하기 위해서 몇 가지 문제점이 있었습니다. 
<br>
<br>
먼저, MLP이 이미지를 처리하기 위해서는 이미지를 1차원 벡터로 평탄화 해야 합니다. 예를 들면, 28x28 픽셀의 흑백 이미지도 784개의 입력 노드를 필요로 합니다. 이렇게 입력 노드가 많아지면, 파라미터가 매우 많아져 학습이 느려지고 과적합의 위험이 커집니다.
<img src="{{'/assets/img/posts/mlp_limit.png' | relative_url}}" alt="MLP 한계점" width="70%" style="display: block; margin: 0 auto;">

뿐만 아니라, 1차원 벡터로 평탄화하는 과정에서 픽셀의 공간적 정보를 손실되게 됩니다. 이로 인해, 학습이 매우 비효율적이고 많은 데이터가 필요하게 됩니다.
아래 그림과 같이, 파란색과 빨간색 데이터는 실제로는 이웃한 데이터이지만, 1차원 벡터로 평탄화하게 되면 28 만큼의 차이가 발생하여 공간적 정보를 잃게 됩니다.
<img src="{{'/assets/img/posts/mlp_limit1.png' | relative_url}}" alt="MLP 한계점" width="40%" style="display: block; margin: 0 auto;">
<br>
<hr class="thin-hr">

#### CNN 모델
이러한 기존의 MLP의 문제점을 해결하기 위해 개발된 모델이 CNN입니다. CNN(Convolutional Neural Network)은 Convolutional Filter를 이용하여 합성곱 연산을 활용한 딥러닝 모델입니다.

<img src="{{"/assets/img/posts/cnn_model_1.png" | relative_url }}" width="600" height="400" alt="CNN 모델" style="display: block; margin: 0 auto;">

CNN은 MLP와 달리 이미지의 특성을 효율적으로 다루기 위해 다음과 같은 구성 요소들을 도입하여 이미지의 공간적 정보를 유지하면서 특징을 추출하는 방식을 사용합니다.
CNN 모델은 크게 두 부분으로 구성이 됩니다.
1. 특징 추출(Feature Extraction)
해당 부분은 주로 여러 개의 Convolutional Layer와 Max Pooling Layer가 반복적되는 형태로 되어있으며, Batch Normalization Layer가 들어가기도 합니다.
보통 이미지의 저수준부터 고수준까지의 특징을 추출하는 역할을 합니다.
<br>
<br>
- 컨볼루션 레이어 (Convolutional Layer)
  * 역할 
  이미지에서 선, 모서리, 질감 등과 같은 다양한 **특징**들을 추출합니다. 이미지의 공간 정보를 보존하면서 특징을 학습하는 것이 강점입니다.
  * 작동 방식
  **필터(Filter)** 또는 **커널(Kernel)**이 이미지 위를 이동하며 특징을 스캔하고 강조합니다. 학습을 통해 필터가 업데이트되어 특징을 잘 포착하게 됩니다.
  * 결과물
  여러 개의 **특징 맵(Feature Map)**을 생성합니다.
  * 특징
  Convolutional Layer의 경우, **Convolutional layer**, **Batch Normalization**, **Activation Function**을 포함하는 경우가 많으며, 각각 특징 추출, 출력 정규화 및 학습 안정화, 비선형성 부여의 역할을 가지고 있습니다.
<br>
<br>
- 풀링 레이어 (Pooling Layer)
  * 역할
  특징 맵의 크기를 줄여(다운샘플링) 계산량을 줄이고, 과적합을 방지하며, 모델이 이미지의 미세한 변화에 덜 민감해지도록(변이 불변성) 만듭니다.
  * 작동 방식 
  주로 **Max Pooling**을 사용해 특정 영역의 가장 큰 값을 선택하여 특징을 압축합니다.
  * 결과물
  크기가 줄어든 새로운 특징 맵을 생성합니다.
<br>
  <br>
2. 분류기(Classifier)
분류기에서는 이미 얻어진 Feature Map을 **Flatten**하여 1차원의 배열로 만들어, 기존의 MLP와 활성화함수를 이용하여 이미지를 분류합니다.
<br>
<br>
- 플래튼 레이어 (Flatten Layer)
  * 역할
  특징 추출 부분에서 생성된 2차원 또는 3차원 특징 맵 데이터를 완전 연결 계층에 입력하기 위해 1차원 배열(벡터)로 변환합니다.
<br>
<br>
- 완전 연결 계층 (Fully Connected Layer, Multi Layer Perceptron, FCL, MLP)
  * 역할
   1차원으로 펼쳐진 특징 벡터를 입력받아 최종 분류를 위한 복잡한 비선형 관계를 학습합니다. 여러 층으로 구성될 수 있습니다.
  * 특징
  MLP와 Activation Function으로 이루어져 있으며, 각각 데이터 분류 및 비선형성 부여의 역할을 한다. 마지막 Layer의 경우, Activation Function을 softmax를 활용하여 각 클래스에 속할 확률을 반환해준다. 
    
<hr class="thick-hr">

## CNN의 발전
<hr class="thick-hr">

### LeNet-5 (1998)
 최초의 성공적인 CNN 모델로써, 핵심 아이디어는 다음과 같습니다. 
- 합성곱(Convolutional) 계층
<br>
이미지의 특징을 추출하는 데 사용됩니다. 필터(커널)가 이미지를 훑으면서 특징 맵(Feature Map)을 생성합니다.
<br>
- 서브샘플링(Subsampling) 계층 (풀링)
<br>
특징 맵의 크기를 줄이고(차원 축소), 중요한 특징만 보존하며, 위치 변화에 덜 민감하게 만듭니다. (평균 풀링 또는 최대 풀링)
<br>
- 지역 수용장(Local Receptive Fields)
<br>
각 뉴런이 이미지의 특정 부분(작은 영역)만 보는 구조입니다. 이는 이미지 전체를 한꺼번에 학습하는 것보다 효율적입니다.
<br>
- 가중치 공유(Shared Weights)
<br>
동일한 필터가 이미지 전체에 적용되므로, 학습해야 할 파라미터 수가 크게 줄어듭니다. 또한, 이미지의 어느 위치에 있든 동일한 특징을 인식할 수 있게 합니다.
<br>
- 다층 구조
<br>
합성곱-풀링 계층을 여러 번 쌓아 점차 고차원의 특징을 추출합니다. 마지막에는 완전 연결 계층(Fully Connected Layer)이 이어집니다.

우편 번호나 은행 수표의 숫자 인식에 성공적으로 적용되어 실용가능성을 입증하였습니다. 뿐만 아니라, 현재까지 발전된 대부분의 CNN 모델이 LeNet의 기본구조(합성곱 - 풀링 - FCL)를 기반으로 발전하였습니다.

<hr class="thin-hr">

### AlexNet (2012)
 LeNet이 MNIST 손글씨 데이터를 기준으로 높은 정확도를 보였지만, 복잡한 이미지 데이터를 다루기에는 부족하였습니다. 이러한 문제점을 해결하기 위해 AlexNet이 등장하였습니다.
AlexNet의 구조는 다음과 같습니다.
<br>
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="{{'/assets/img/posts/alexnet.png' | relative_url }}" alt="alexnet 모델" width="500">
  <img src="{{'/assets/img/posts/alexnet1.png' | relative_url }}" alt="alexnet 모델" width="400">
</div>
<br>
이러한 AlexNet은 다음과 같은 특징이 있습니다.
- 깊은 네트워크
<br>
8개의 계층 (5개의 합성곱 계층, 3개의 완전 연결 계층)으로 LeNet보다 훨씬 깊고 복잡합니다.
<br>
- ReLU(Rectified Linear Unit) 활성화 함수
<br>
기존의 Sigmoid나 tanh 함수보다 훨씬 빠르게 수렴하고, 기울기 소실(Vanishing Gradient) 문제를 완화하여 깊은 네트워크의 학습을 가능하게 했습니다. (음수는 0, 양수는 그대로 통과)
<br>
- 드롭아웃(Dropout)
<br>
훈련 중 무작위로 뉴런의 일부를 비활성화하여 과적합(Overfitting)을 방지하는 정규화(Regularization) 기법을 도입했습니다.
<br>
- GPU 활용
<br>
대량의 연산이 필요한 딥러닝 모델 학습을 위해 두 개의 GPU를 사용하여 병렬 연산을 수행했습니다. 이는 대규모 모델 학습의 가능성을 열었습니다.
<br>
- 데이터 증강(Data Augmentation)
<br>
이미지 크기, 회전, 밝기 등을 인위적으로 변경하여 훈련 데이터의 양을 늘려 과적합을 줄이고 모델의 일반화 성능을 높였습니다.

AlexNet은 대규모 데이터셋과 GPU의 병렬연산을 통해 복잡한 이미지 데이터를 처리하는 데에 뛰어난 성능을 보일 수 있음을 증명했습니다.

<hr class="thin-hr">

### VGGNet
AlexNet이 이미지 처리에 뛰어난 성능을 보였지만, 구조가 단순했습니다. 이러한 상황에서 **더 깊은 네트워크** 즉, depth가 증가함에 따른 성능의 변화를 실험하고자 했습니다.

VGGNet의 구조는 다음과 같습니다.

<img src="{{"/assets/img/posts/vggnet.png" | relative_url }}" width="600" height="400" alt="CNN 모델" style="display: block; margin: 0 auto;">

VGGNet은 depth의 증가를 위해 다음과 같은 특징을 추가하였습니다.
- 작은 필터의 반복
모든 합성곱 계층에서 오직 3x3 크기의 필터만 사용했습니다. 예를 들어, 3x3 필터 두 번을 연속으로 사용하는 것은 5x5 필터 한 번과 동일한 수용 영역(Receptive Field)을 가지면서도 파라미터 수는 더 적고, 비선형성을 두 번 추가하여 모델의 표현력을 높이는 효과가 있습니다.
- 깊은 구조
16개 또는 19개 계층으로 이루어져 매우 깊은 네트워크를 구성했습니다. 이는 네트워크가 깊어질수록 더 추상적이고 복잡한 특징을 학습할 수 있음을 보여주었습니다.
- 균일한 아키텍처
AlexNet보다 훨씬 규칙적이고 반복적인 구조를 가져, 모델 설계와 확장이 직관적이었습니다.

<hr class="thin-hr">

### GoogLeNet(Inception v1)
핵심 아이디어 및 구조: 구글이 개발하여 2014년 ILSVRC에서 우승했습니다. VGGNet이 단순히 깊이를 늘린 것과 달리, 이 모델은 '효율성'과 '모듈화'에 초점을 맞췄습니다.
인셉션(Inception) 모듈: 이 모델의 핵심입니다. 하나의 모듈 내에서 **다양한 크기의 필터(1x1, 3x3, 5x5 합성곱)와 풀링(3x3 Max Pooling)**을 병렬적으로 적용한 후, 그 결과물들을 하나의 깊이 차원으로 **연결(Concatenate)**합니다. 이는 네트워크가 동시에 다양한 스케일의 특징을 학습하고, 최적의 특징 조합을 스스로 찾아내도록 합니다.
1x1 합성곱의 활용: 인셉션 모듈 내부에서 1x1 합성곱 필터를 사용하여 **차원 축소(Dimension Reduction)**를 수행했습니다. 예를 들어, 5x5 합성곱을 수행하기 전에 1x1 합성곱으로 채널 수를 줄이면, 전체 연산량을 크게 줄일 수 있습니다. 이는 "병목 계층(Bottleneck Layer)"이라고도 불립니다.
희소성(Sparsity) 원칙: 효율적인 신경망은 대부분의 뉴런이 동시에 활성화되는 것이 아니라, 특정 특징에 반응하는 일부 뉴런만 활성화되는 '희소성'을 갖는다는 아이디어에서 출발했습니다.
보조 분류기(Auxiliary Classifier): 깊은 네트워크에서 기울기 소실 문제를 완화하기 위해 네트워크 중간에도 분류기를 두어 역전파 시 기울기를 보강하는 역할을 했습니다.
의의: 무작정 네트워크를 깊고 넓게 만드는 것이 아니라, 모듈화된 효율적인 구조를 통해 파라미터 수를 줄이면서도 성능을 향상시킬 수 있음을 보여주었습니다. 이는 효율적인 모델 설계의 중요성을 강조했습니다.

<hr class="thin-hr">

### ResNet

핵심 아이디어 및 구조: 카이밍 허(Kaiming He) 등이 개발하여 2015년 ILSVRC에서 우승하며 딥러닝 연구에 혁신을 가져왔습니다. VGGNet처럼 네트워크를 매우 깊게 쌓았을 때 발생하는 고질적인 문제인 **'퇴화(Degradation)' 문제(네트워크가 깊어질수록 성능이 오히려 저하되는 현상)**를 해결했습니다.
잔차 연결(Residual Connection) 또는 스킵 연결(Skip Connection): 이 모델의 가장 핵심적인 개념입니다.
일반적인 네트워크 계층은 입력 x로부터 H(x)라는 매핑을 학습합니다. ResNet은 이 대신 F(x) = H(x) - x, 즉 **잔차(Residual)**를 학습합니다.
그러면 계층의 최종 출력은 H(x) = F(x) + x가 됩니다. 여기서 x는 합성곱 계층을 거치지 않고 바로 다음 계층으로 전달되는 '지름길(Identity Mapping)'입니다.
이 F(x) + x 구조는 네트워크가 H(x)를 직접 학습하는 것보다 F(x)를 학습하는 것이 더 쉽다는 아이디어에서 출발합니다. 만약 H(x)가 x와 큰 차이가 없다면, F(x)를 0으로 만들면 됩니다. 0을 학습하는 것은 기존 계층이 x를 완벽하게 모사하는 것보다 훨씬 쉽습니다.
기울기 소실 완화: 잔차 연결은 역전파 시 기울기가 x 경로를 통해 직접 전달될 수 있게 하여, 기울기 소실 문제를 크게 완화하고 매우 깊은 네트워크도 효과적으로 학습할 수 있게 했습니다.
초고심층 신경망: 이 구조 덕분에 152개 층, 심지어 1000개 층이 넘는 네트워크도 안정적으로 학습할 수 있게 되었습니다.
의의: 딥러닝 모델의 깊이 한계를 근본적으로 해결하여, 더욱 복잡하고 강력한 특징 추출 능력을 가능하게 했습니다. 잔차 연결 개념은 이후 모든 딥러닝 아키텍처에 광범위하게 적용되며 현대 딥러닝의 가장 중요한 구성 요소 중 하나가 되었습니다.