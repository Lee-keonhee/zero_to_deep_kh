# ZtoD 블로그 - 간소화 완료

블로그 중심으로 불필요한 기능들을 제거하고 핵심 기능만 남겼습니다.

## 제거된 기능

- ❌ Gallery 페이지 및 관련 기능
- ❌ Shop/Product 기능
- ❌ Contact 폼
- ❌ Newsletter 구독
- ❌ 복잡한 프로필 설정 (work experience, education, projects)
- ❌ Snipcart (쇼핑 기능)
- ❌ 소셜 미디어 공유 버튼
- ❌ Disqus/Hyvor 댓글 시스템
- ❌ 검색 기능

## 유지된 핵심 기능

- ✅ 블로그 포스팅
- ✅ 다크모드
- ✅ 이미지 및 섬네일
- ✅ MathJax 수식 렌더링
- ✅ 코드 하이라이팅
- ✅ 카테고리 및 태그
- ✅ 페이지네이션
- ✅ RSS Feed
- ✅ SEO 최적화
- ✅ 반응형 디자인

## 디렉토리 구조

```
ZtoD/
├── _config.yml           # 간소화된 설정
├── _posts/               # 블로그 포스트
│   └── 2025-01-29-propagation.md
├── _layouts/             # (필요시 추가)
├── _includes/            # 재사용 컴포넌트
│   └── header.html
├── _sass/                # 스타일시트
├── assets/
│   └── img/
│       └── posts/        # 포스트 이미지
├── blog/
│   └── index.html
├── about.md
├── index.html
├── Gemfile
├── .gitignore
└── README.md
```

## 다음 단계

1. **스타일시트 추가**: `_sass/` 디렉토리에 다크모드 스타일 추가
2. **레이아웃 템플릿**: `_layouts/` 디렉토리에 post, blog, home 레이아웃 추가
3. **이미지 추가**: `assets/img/` 에 로고, 커버 이미지 등 추가
4. **더 많은 포스트**: `_posts/` 에 추가 글 작성

## 로컬 테스트

```bash
bundle install
bundle exec jekyll serve
```

브라우저에서 `http://localhost:4000/zero_to_deep_kh` 접속

## GitHub Pages 배포

1. GitHub에 푸시
2. Settings > Pages 에서 branch 선택
3. 자동 배포 완료
