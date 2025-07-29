---
layout: post
title:  "순전파/ 역전파 예제로 이해하기"
summary: "순전파와 역전파의 계산을 통해, 가중치의 최적화 과정 확인"
author: Keonhee
date: '2025-07-29 16:20:00 +0900'
category: 순전파와 역전파
#thumbnail: /assets/img/posts/propagation1.png
keywords: 딥러닝
permalink: /blog/propagation/
usemathjax: true
---

![순전파 역전파 그림]('/assets/img/posts/propagation1.png'){: width="50%"}

초기값:
<div style="text-align: left;">
\( X = \begin{pmatrix} x_1 & x_2 \end{pmatrix} = \begin{pmatrix} 0.5 & 0.3 \end{pmatrix},\quad W = \begin{pmatrix} w_1 & w_2 \\ w_3 & w_4 \\ w_5 & w_6 \end{pmatrix} = \begin{pmatrix} 0.3 & 0.2 \\ 0.2 & 0.3 \\ 0.4 & 0.5 \end{pmatrix},\quad y_1 = 0.7 \)
</div>

<br>

<hr class="thick-hr">

#### 순전파(Forward Propagation)

<hr class="thick-hr">

##### 가중합 계산
<div style="margin-top: 10px;"></div>
<div style="text-align: left;">
\( z_1 = w_1 \cdot x_1 + w_3 \cdot x_2 \)
<br>
<div style="margin-top: 15px;"></div>

\( z_2 = w_2 \cdot x_1 + w_4 \cdot x_2 \)
</div>
<br>

<hr class="thin-hr">

##### 활성화 함수 계산
<div style="margin-top: 15px;"></div>
<div style="text-align: left;color: white;">
\( a_1 = Sigmoid(z_1) = \frac{1}{1+e^{-z_1}} = \frac{1}{1+e^{-0.21}} = 0.552 \)
<br>
<div style="margin-top: 10px;"></div>

\( a_2 = Sigmoid(z_1) =  \frac{1}{1+e^{-z_2}} = \frac{1}{1+e^{-0.19}} = 0.547 \)
</div>
<br>

<hr class="thin-hr">

##### 결과
<div style="margin-top: 15px;"></div>

<div style="text-align: left;">
\( \hat{y_1} = w_5 \cdot a_1 + w_6 \cdot a_2 = 0.552 \times 0.4 + 0.547 \times 0.5 = 0.4943 \)
</div>

<br>

<hr class="thin-hr">

##### 오차 계산 (Mean-Squred Error, MSE)
<div style="margin-top: 15px;"></div>

실제 정답 $$y_1 = 0.7$$일 때, 모델의 예측값 $$\hat{y} = 0.4943$$ 이므로 손실은 다음과 같습니다.
<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">

$$
\begin{aligned}
L &= (y_1 - \hat{y})^2 \\
  &= (0.7 - 0.4943)^2 = (0.2057)^2 \approx 0.04231
\end{aligned}
$$
</div>

<br>


<hr class="thick-hr">

#### 역전파

<hr class="thick-hr">

순전파를 통해 얻은 예측값($$\hat{y}$$)이 실제 정답($$y_1$$)과 얼마나 다른지 **오차(손실 $$L$$)를 계산**했습니다. 이제 이 오차를 줄이기 위해 모델의 가중치($$w_1$$부터 $$w_6$$)들을 업데이트해야 합니다.

**역전파는 이 오차가 각 가중치에 어떤 영향을 미쳤는지를 파악하여 가중치들을 조정하는 과정**입니다. 손실 함수 $$L$$을 각 가중치 $$w_i$$에 대해 **미분(기울기 $$\frac{\partial L}{\partial w_i}$$)**하여, 가중치를 어떤 방향으로 얼마나 조절해야 오차가 최소화되는지를 찾아냅니다. 이때, 복잡한 신경망 구조 때문에 **연쇄 법칙(Chain Rule)**을 사용하여 출력층부터 입력층 방향으로 효율적으로 기울기를 계산하게 됩니다.

[//]: # (<div style="text-align: left;color: white;">)

<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">

$$
\begin{aligned}
\frac{\partial L}{\partial w_5}
&= \frac{\partial L}{\partial \hat{y_1}} \cdot \frac{\partial \hat{y_1}}{\partial w_5} \\[10pt]
&= \frac{\partial ((y_1 - \hat{y_1})^2)}{\partial \hat{y_1}} \cdot \frac{\partial (w_5 \cdot a_1 + w_6 \cdot a_2)}{\partial w_5} \\[10pt]
&= -2(y_1 - \hat{y_1}) \cdot a_1 \\[10pt]
&= -2 \cdot (0.7 - 0.4943) \cdot 0.552 \\[10pt]
&= -0.227
\end{aligned}
$$
</div>

<br>

<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">

$$
\begin{aligned}
\frac{\partial L}{\partial w_6}
&= \frac{\partial L}{\partial \hat{y_1}} \cdot \frac{\partial \hat{y_1}}{\partial w_6} \\[10pt]
&= \frac{\partial (y_1 - \hat{y_1})^2}{\partial \hat{y_1}} \cdot \frac{\partial (w_5 \cdot a_1 + w_6 \cdot a_2)}{\partial w_6} \\[10pt]
&= -2(y_1 - \hat{y_1}) \cdot a_2 \\[10pt]
&= -2 \cdot (0.7 - 0.4943) \cdot 0.547 \\[10pt]
&= -0.225
\end{aligned}
$$
</div>

<br>

<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">

$$
\begin{aligned}
\frac{\partial L}{\partial w_1}
&= \frac{\partial L}{\partial \hat{y_1}} \cdot \frac{\partial \hat{y_1}}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1} \\[10pt]
&= \frac{\partial (y_1 - \hat{y_1})^2}{\partial \hat{y_1}} \cdot \frac{\partial (w_5 \cdot a_1 + w_6 \cdot a_2)}{\partial a_1} \cdot \frac{\partial \left(\frac{1}{1+e^{-z_1}}\right)}{\partial z_1} \cdot \frac{\partial (w_1 \cdot x_1 + w_3 \cdot x_2)}{\partial w_1} \\[10pt]
&= -2(y_1 - \hat{y_1}) \cdot w_5 \cdot \left(\frac{1}{1+e^{-z_1}}\right) \cdot \left(1 - \frac{1}{1+e^{-z_1}}\right) \cdot x_1 \\[10pt]
&= -0.0204
\end{aligned}
$$

</div>

<br>

이와 마찬가지로,
<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">

$$
\begin{aligned}
\frac{\partial L}{\partial w_2} &= -0.0255 \\[10pt]
\frac{\partial L}{\partial w_3} &= -0.0122 \\[10pt]
\frac{\partial L}{\partial w_4} &= -0.0153
\end{aligned}
$$
</div>

<hr class="thin-hr">

#### 파라미터 업데이트 (가중치 조정)
<div style="margin-top: 15px;"></div>

학습률($$\eta$$)을 0.1이라고 할 때, 각 가중치는 다음 공식에 따라 업데이트됩니다.
<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$$
</div>
해당 공식에 따라 각 가중치를 업데이트를 하게 되면 다음과 같은 결과를 얻을 수 있습니다.

<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">
$$
w_1' = w_1 - \eta \frac{\partial L}{\partial w_1} = 0.3 - (-0.0204) \approx 0.302
$$
</div>

<div style="width: fit-content; margin-left: 0; margin-right: auto; color: white;">
$$
\begin{aligned}
w_2' = 0.203, \quad
w_3' = 0.201, \quad
w_4' = 0.302, \quad
w_5' = 0.423, \quad
w_6' = 0.523
\end{aligned}
$$
</div>
<br>

<hr class="thin-hr">

#### 업데이트된 가중치로 새로운 예측값 계산
<div style="margin-top: 15px;"></div>

업데이트된 가중치로 다시 계산해 보면, 예측값 $$\hat{y_1}$$ 는 다음과 같이 변합니다.

<div style="text-align: left;">
\( \hat{y_1} = 0.5199 \)
</div>
<div style="margin-top: 10px;"></div>

이전 예측 값($$0.4943$$)보다 실제 정답($$0.7$$)에 더 가까워지며 오차가 줄어들었음을 확인할 수 있습니다. 이처럼 신경망은 오차를 최소화하는 방향으로 반복적인 학습을 수행합니다.

<hr class="thin-hr">
