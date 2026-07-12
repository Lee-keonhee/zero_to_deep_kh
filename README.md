9# ZtoD (Zero to Deep) Blog

AI의 기초부터 응용까지 함께 배우는 딥러닝 블로그입니다.

## 주요 기능

- ✍️ 블로그 포스팅 중심
- 🌙 다크모드 지원
- 🖼️ 이미지 및 섬네일 지원
- 📊 수식 렌더링 (MathJax)
- 💻 코드 하이라이팅
- 📱 반응형 디자인

## 로컬 실행 방법

### 필수 요구사항

- Ruby (>= 2.7.0)
- Bundler
- Jekyll

### 설치 및 실행

```bash
# 저장소 클론
git clone https://github.com/Lee-keonhee/zero_to_deep_kh.git
cd zero_to_deep_kh

# 의존성 설치
bundle install

# 로컬 서버 실행
bundle exec jekyll serve

# 브라우저에서 http://localhost:4000 접속
```

## 포스트 작성 방법

`_posts` 디렉토리에 다음 형식으로 파일을 생성합니다:

```markdown
---
layout: post
title: 제목
summary: 요약
author: keonhee
date: YYYY-MM-DD HH:MM:SS +0900
category: 카테고리
keywords: 키워드1, 키워드2
permalink: /blog/포스트명/
usemathjax: true
thumbnail: /assets/img/posts/이미지.png
---

본문 내용...
```

## 디렉토리 구조

```
.
├── _config.yml          # Jekyll 설정
├── _posts/              # 블로그 포스트
├── _layouts/            # 레이아웃 템플릿
├── _includes/           # 재사용 컴포넌트
├── _sass/               # 스타일시트
├── assets/
│   └── img/
│       └── posts/       # 포스트 이미지
├── blog/                # 블로그 페이지
└── about.md             # About 페이지
```

## 사용 기술

- Jekyll 4.3.3
- Bootstrap
- Font Awesome
- MathJax (수식 렌더링)
- Rouge (코드 하이라이팅)

## 라이선스

MIT License


병신ㅋ 나라 망하는게 지금 지역 성별 등 여러가지 나누고 있는게 문제인데 이걸또 지역갈등 만드네요ㅎ 몇천 몇억.. 비교도 안되는돈이 세어나가고있는데ㅋㅋ 이러면 이제 저는 신고 당해서 답변도 못달겠죠 수고 하세요