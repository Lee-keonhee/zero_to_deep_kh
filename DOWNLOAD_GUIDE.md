# 📦 다운로드 받아야 할 파일 목록

## 현재 생성된 파일 구조

```
zero_to_deep_kh/
├── _config.yml              ⭐ Jekyll 설정 파일
├── Gemfile                  ⭐ Ruby 의존성 파일
├── README.md                📝 프로젝트 설명
├── .gitignore               🔒 Git 제외 파일
│
├── _includes/               📁 재사용 컴포넌트
│   ├── header.html
│   └── post_metadata.html
│
├── _layouts/                📁 페이지 레이아웃 템플릿
│   ├── category.html
│   └── tag.html
│
├── _plugins/                📁 Jekyll 플러그인
│   └── category_tag_generator.rb
│
├── _posts/                  📁 블로그 포스트 (마크다운)
│   ├── 2025-01-15-pytorch-getting-started.md
│   ├── 2025-01-20-cnn-basics.md
│   └── 2025-01-29-propagation.md
│
├── _sass/                   📁 스타일시트 (아직 비어있음)
│
├── assets/                  📁 이미지 및 정적 파일
│   └── img/
│       └── posts/           (여기에 포스트 이미지 넣기)
│
├── blog/                    📁 블로그 메인 페이지
│   └── index.html
│
├── about.md                 📄 소개 페이지
├── index.html               📄 홈페이지
├── categories.html          📄 카테고리 목록
├── tags.html                📄 태그 목록
│
└── 문서들/                  📚 가이드 문서
    ├── CATEGORY_TAG_GUIDE.md
    └── PROJECT_SUMMARY.md
```

## ⚠️ 아직 추가 안된 중요한 파일들

다음 파일들은 원본 테마에서 가져와야 합니다:

### 1. 레이아웃 파일 (`_layouts/`)
- `default.html` - 기본 레이아웃
- `post.html` - 포스트 레이아웃  
- `blog.html` - 블로그 목록 레이아웃
- `home.html` - 홈페이지 레이아웃
- `about-me.html` - About 페이지 레이아웃

### 2. Include 파일 (`_includes/`)
- `head.html` - HTML head 태그
- `footer.html` - 푸터
- `hero.html` - 홈페이지 히어로 섹션
- `recent_articles.html` - 최근 글 목록

### 3. 스타일시트 (`_sass/`)
- `_devlopr.scss` - 메인 스타일
- 다크모드 스타일

### 4. Assets (`assets/`)
- `css/main.scss` - CSS 진입점
- `js/` - JavaScript 파일들
- `img/` - 이미지 파일들
  - `page_logo.png` - 로고
  - `sample_cover.jpg` - 커버 이미지

## 🚀 빠른 시작 (2가지 방법)

### 방법 1: 지금 생성된 파일만으로 시작

1. **현재 outputs 폴더의 모든 파일 다운로드**
2. **원본 테마에서 필요한 파일만 복사**:
   - `_layouts/` 폴더 전체
   - `_includes/` 폴더의 나머지 파일들
   - `_sass/` 폴더 전체
   - `assets/` 폴더 전체

### 방법 2: 원본 테마 전체를 클론 후 수정

```bash
# 1. 원본 테마 클론
git clone https://github.com/sujaykundu777/devlopr-jekyll.git zero_to_deep_kh

# 2. 불필요한 파일 삭제
cd zero_to_deep_kh
rm -rf gallery/ shop/ contact.md _products/

# 3. 제가 생성한 파일들로 교체
# (_config.yml, _posts/, _plugins/, categories.html, tags.html 등)
```

## 📥 다운로드 방법

### Claude에서 다운로드

현재 채팅에서 제가 생성한 파일들의 링크를 제공해드렸습니다.
아래 파일들을 순서대로 다운로드 받으세요:

1. **필수 설정 파일**
   - `_config.yml`
   - `Gemfile`
   - `.gitignore`

2. **컨텐츠 파일**
   - `_posts/` 폴더의 모든 `.md` 파일
   - `about.md`
   - `index.html`
   - `categories.html`
   - `tags.html`

3. **템플릿 파일**
   - `_layouts/` 폴더
   - `_includes/` 폴더
   - `_plugins/` 폴더

4. **블로그 폴더**
   - `blog/index.html`

## 🔧 설치 후 실행

```bash
# 1. 폴더로 이동
cd zero_to_deep_kh

# 2. 의존성 설치
bundle install

# 3. 로컬 서버 실행
bundle exec jekyll serve

# 4. 브라우저에서 확인
# http://localhost:4000/zero_to_deep_kh
```

## ❓ 어떤 방법을 추천하나요?

**초보자**: 방법 2 (원본 클론 후 수정)
- 모든 파일이 갖춰져 있어 바로 실행 가능
- 필요 없는 것만 삭제하면 됨

**경험자**: 방법 1 (필요한 것만 조합)
- 깔끔하게 시작 가능
- 원하는 대로 커스터마이징

## 📝 다음 단계

파일을 다운로드 받은 후:
1. Ruby 설치
2. `bundle install`
3. 로컬에서 실행 테스트
4. 포스트 작성 시작!
