---
layout: post
title:  "CNN의 발전 과정"
summary: "LeNet 부터 ResNet까지"
author: keonhee
date: '2025-08-19 12:20:00 +0900'
category: Deeplearning, CNN
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/Development_of_CNN/
usemathjax: true
---
<hr class="thick-hr">

### CNN

<hr class="thick-hr">

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

<img src="{{"/assets/img/posts/cnn_model_1.png" | relative_url }}" width="600" height="400" alt="CNN 모델">

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
- 서브샘플링(Subsampling) 계층 (풀링)
<br>
특징 맵의 크기를 줄이고(차원 축소), 중요한 특징만 보존하며, 위치 변화에 덜 민감하게 만듭니다. (평균 풀링 또는 최대 풀링)
- 지역 수용장(Local Receptive Fields)
<br>
각 뉴런이 이미지의 특정 부분(작은 영역)만 보는 구조입니다. 이는 이미지 전체를 한꺼번에 학습하는 것보다 효율적입니다.
- 가중치 공유(Shared Weights)
<br>
동일한 필터가 이미지 전체에 적용되므로, 학습해야 할 파라미터 수가 크게 줄어듭니다. 또한, 이미지의 어느 위치에 있든 동일한 특징을 인식할 수 있게 합니다.
- 다층 구조
<br>
합성곱-풀링 계층을 여러 번 쌓아 점차 고차원의 특징을 추출합니다. 마지막에는 완전 연결 계층(Fully Connected Layer)이 이어집니다.

우편 번호나 은행 수표의 숫자 인식에 성공적으로 적용되어 실용가능성을 입증하였습니다. 뿐만 아니라, 현재까지 발전된 대부분의 CNN 모델이 LeNet의 기본구조(합성곱 - 풀링 - FCL)를 기반으로 발전하였습니다.

<hr class="thin-hr">

### AlexNet (2012)
 LeNet이 MNIST 손글씨 데이터를 기준으로 높은 정확도를 보였지만, 복잡한 이미지 데이터를 다루기에는 부족하였습니다. 이러한 문제점을 해결하기 위해 AlexNet이 등장하였습니다.
AlexNet의 구조는 다음과 같습니다.
<img src="{{"/assets/img/posts/alexnet.png" | relative_url }}" width="400" height="300" alt="alexnet 모델">
<img src="{{"/assets/img/posts/alexnet1.png" | relative_url }}" width="400" height="300" alt="alexnet 모델">

이러한 AlexNet은 다음과 같은 특징이 있습니다.
- 깊은 네트워크
<br>
8개의 계층 (5개의 합성곱 계층, 3개의 완전 연결 계층)으로 LeNet보다 훨씬 깊고 복잡합니다.
- ReLU(Rectified Linear Unit) 활성화 함수
<br>
기존의 Sigmoid나 tanh 함수보다 훨씬 빠르게 수렴하고, 기울기 소실(Vanishing Gradient) 문제를 완화하여 깊은 네트워크의 학습을 가능하게 했습니다. (음수는 0, 양수는 그대로 통과)
- 드롭아웃(Dropout)
<br>
훈련 중 무작위로 뉴런의 일부를 비활성화하여 과적합(Overfitting)을 방지하는 정규화(Regularization) 기법을 도입했습니다.
- GPU 활용
<br>
대량의 연산이 필요한 딥러닝 모델 학습을 위해 두 개의 GPU를 사용하여 병렬 연산을 수행했습니다. 이는 대규모 모델 학습의 가능성을 열었습니다.
- 데이터 증강(Data Augmentation)
<br>
이미지 크기, 회전, 밝기 등을 인위적으로 변경하여 훈련 데이터의 양을 늘려 과적합을 줄이고 모델의 일반화 성능을 높였습니다.

AlexNet은 대규모 데이터셋과 GPU의 병렬연산을 통해 복잡한 이미지 데이터를 처리하는 데에 뛰어난 성능을 보일 수 있음을 증명했습니다.

