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
CNN은 MLP와 달리 이미지의 특성을 효율적으로 다루기 위해 다음과 같은 구성요소들을 도입하여 이미지의 공간적 정보를 유지하면서 특징을 추출하는 방식을 사용합니다.

1. 합성곱 계층 (Convolutional Layer)
- **합성곱 필터(Convolutional Filter)**를 활용하여 이미지 전체를 훑으며 동일한 필터 가중치(Shared Weights)를 공유합니다.이를 통해 학습해야 할 파라미터의 수가 획기적으로 줄어듭니다.
- 또한, 동일한 특징을 이미지 내 어떤 위치에서든 효과적으로 감지할 수 있는 '위치 불변성'을 확보합니다.
- 이미지를 1차원으로 평탄화하지 않고 직접 처리함에 따라 픽셀 간의 공간적 관계를 보존하면서 특징 맵(Feature Map)을 생성합니다.
2. 풀링 계층 (Pooling Layer)
- 합성곱 계층에서 생성된 특징 맵의 **크기(차원)**를 줄여 모델의 복잡도와 연산량을 감소시킵니다.
- 특징의 대략적인 위치 정보를 유지하면서 미세한 위치 변화에도 강건한(강한) 변화 불변성을 확보하여 모델의 일반화 성능을 향상시킵니다. 

이 외에도 CNN은 여러 계층을 깊게 쌓아 올림으로써 저수준의 특징부터 고수준의 의미 있는 특징까지 계층적으로 학습할 수 있는 능력을 갖추게 됩니다.

