# 카테고리와 태그 자동 생성 가이드

## 개요

이 시스템은 `_posts` 폴더의 모든 포스트에서 `category`와 `tags`를 자동으로 읽어서 해당 페이지를 생성합니다.

## 작동 방식

### 1. 포스트 작성 시

```yaml
---
layout: post
title: 내 포스트 제목
category: 딥러닝기초           # 단일 카테고리
tags: [PyTorch, CNN, 이미지]  # 여러 태그
---
```

### 2. 자동 생성되는 페이지

Jekyll 빌드 시 자동으로 다음 페이지들이 생성됩니다:

- `/category/딥러닝기초/` - "딥러닝기초" 카테고리의 모든 포스트
- `/tag/pytorch/` - "PyTorch" 태그의 모든 포스트
- `/tag/cnn/` - "CNN" 태그의 모든 포스트
- `/tag/이미지/` - "이미지" 태그의 모든 포스트

### 3. 전체 목록 페이지

- `/categories/` - 모든 카테고리 목록 및 포스트 수
- `/tags/` - 모든 태그 클라우드 및 포스트 수

## 사용 예제

### 예제 1: 기본 사용

```yaml
---
layout: post
title: PyTorch 시작하기
category: 프레임워크
tags: [PyTorch, 딥러닝, Python]
---
```

생성되는 URL:
- `/category/프레임워크/`
- `/tag/pytorch/`
- `/tag/딥러닝/`
- `/tag/python/`

### 예제 2: 여러 포스트, 같은 카테고리

```yaml
# 포스트 1
---
category: 딥러닝기초
tags: [순전파, 역전파]
---

# 포스트 2
---
category: 딥러닝기초
tags: [CNN, 이미지처리]
---
```

`/category/딥러닝기초/`에 두 포스트가 모두 표시됩니다.

### 예제 3: 한글/영문 혼용

```yaml
---
category: Machine Learning
tags: [머신러닝, ML, 알고리즘]
---
```

모두 자동으로 URL-safe한 slug로 변환됩니다:
- `/category/machine-learning/`
- `/tag/머신러닝/`
- `/tag/ml/`

## 포스트에 카테고리/태그 표시

포스트 레이아웃에 다음을 추가하면 카테고리와 태그가 표시됩니다:

```liquid
{% include post_metadata.html %}
```

## 네비게이션에 추가

메인 메뉴에 카테고리/태그 페이지를 추가하려면 `_config.yml`에:

```yaml
urls:
    - text: Home
      url: /
    - text: Blog
      url: /blog
    - text: Categories
      url: /categories
    - text: Tags
      url: /tags
```

## 현재 생성된 구조

```
사이트/
├── category/
│   ├── 딥러닝기초/
│   └── 프레임워크/
├── tag/
│   ├── pytorch/
│   ├── cnn/
│   ├── 이미지처리/
│   └── ...
├── categories/      # 전체 카테고리 목록
└── tags/           # 전체 태그 목록
```

## 장점

1. **자동화**: 포스트만 작성하면 페이지가 자동 생성
2. **유연성**: 카테고리/태그 개수 제한 없음
3. **한글 지원**: 한글 카테고리/태그 완벽 지원
4. **SEO**: 각 카테고리/태그마다 개별 URL
5. **네비게이션**: 사용자가 쉽게 관련 포스트 탐색

## 주의사항

- `category`는 단일 값 (하나의 카테고리만)
- `tags`는 배열 (여러 태그 가능)
- 카테고리/태그 이름에 특수문자 사용 시 URL이 자동으로 정리됨
- 빌드 시 `_site/category/` 와 `_site/tag/` 폴더에 생성됨

## 트러블슈팅

### 페이지가 생성되지 않는 경우

1. `_plugins` 폴더에 `category_tag_generator.rb`가 있는지 확인
2. `_layouts/category.html`과 `_layouts/tag.html`이 있는지 확인
3. Jekyll을 재시작: `bundle exec jekyll serve --force_polling`

### GitHub Pages에서 작동하지 않는 경우

GitHub Pages는 커스텀 플러그인을 지원하지 않습니다. 
대안:
1. GitHub Actions를 사용하여 빌드
2. 또는 카테고리/태그 페이지를 수동으로 생성

## 수동 생성 방법 (GitHub Pages용)

각 카테고리마다 수동으로 파일 생성:

```
category/딥러닝기초.md:
---
layout: category
category: 딥러닝기초
---
```

이 경우 새 카테고리 추가 시 파일을 직접 생성해야 합니다.
