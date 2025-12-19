---
layout: post
title: PyTorch 시작하기
summary: PyTorch 설치부터 기본 텐서 연산까지
author: keonhee
date: 2025-01-15 10:00:00 +0900
category: 프레임워크
tags: [PyTorch, 딥러닝, Python]
keywords: PyTorch, 텐서, 딥러닝 프레임워크
permalink: /blog/pytorch-getting-started/
usemathjax: true
thumbnail: /assets/img/posts/pytorch_thumbnail.png
---

# PyTorch 시작하기

PyTorch는 Facebook AI Research에서 개발한 오픈소스 딥러닝 프레임워크입니다.

## 설치

```bash
# CPU 버전
pip install torch torchvision

# CUDA 11.8 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 기본 텐서 연산

```python
import torch

# 텐서 생성
x = torch.tensor([[1, 2], [3, 4]])
print(x)

# 랜덤 텐서
y = torch.randn(2, 2)
print(y)

# 연산
z = x + y
print(z)

# 행렬 곱셈
result = torch.matmul(x, y)
print(result)
```

## Autograd

PyTorch의 자동 미분 기능:

```python
# requires_grad=True로 설정
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# 역전파
y.backward()

# 그래디언트 확인
print(x.grad)  # dy/dx = 2x = 4.0
```

## 간단한 신경망

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
print(model)
```

## PyTorch vs TensorFlow

| 특징 | PyTorch | TensorFlow |
|------|---------|------------|
| 정의 방식 | Dynamic | Static/Dynamic |
| 디버깅 | 쉬움 | 보통 |
| 배포 | 보통 | 쉬움 |

## 다음 단계

- 데이터 로더 사용법
- 학습 루프 작성
- 모델 저장 및 로드
