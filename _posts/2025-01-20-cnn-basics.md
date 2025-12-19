---
layout: post
title: CNN(Convolutional Neural Network) 이해하기
summary: 합성곱 신경망의 구조와 동작 원리를 알아봅니다
author: keonhee
date: 2025-01-20 14:30:00 +0900
category: 딥러닝기초
tags: [CNN, 이미지처리, 신경망, 합성곱]
keywords: CNN, Convolutional Neural Network, 합성곱 신경망, 딥러닝
permalink: /blog/cnn-basics/
usemathjax: true
thumbnail: /assets/img/posts/cnn_thumbnail.png
---

# CNN(Convolutional Neural Network) 기초

합성곱 신경망(CNN)은 이미지 처리에 특화된 딥러닝 모델입니다.

## CNN의 주요 구성 요소

### 1. 합성곱 층 (Convolutional Layer)

```python
import torch.nn as nn

conv_layer = nn.Conv2d(
    in_channels=3,    # RGB 이미지
    out_channels=64,  # 64개의 필터
    kernel_size=3,    # 3x3 커널
    padding=1
)
```

### 2. 풀링 층 (Pooling Layer)

풀링은 특징 맵의 크기를 줄이는 역할을 합니다.

$$
output_{size} = \frac{input_{size} - kernel_{size}}{stride} + 1
$$

### 3. 완전 연결 층 (Fully Connected Layer)

최종 분류를 수행합니다.

## 간단한 CNN 구현

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 생성
model = SimpleCNN()
print(model)
```

## CNN의 장점

1. **파라미터 공유**: 같은 필터를 이미지 전체에 사용
2. **위치 불변성**: 객체의 위치가 달라도 인식 가능
3. **계층적 특징 학습**: 저수준부터 고수준 특징까지 학습

## 다음 단계

- ResNet, VGG 같은 고급 아키텍처 학습
- Transfer Learning 적용
- 실제 이미지 분류 프로젝트
