---
layout: post
title: MCP(Model Context Protocol) -API-통합
summary: MCP(Model Context Protocol) -API-통합
author: keonhee
date: 2025-12-19 10:00:00 +0900
category: MCP
keywords: MCP
permalink: /blog/MCP_5/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
imageNameKey: MCP
---

# 실제 API를 활용한 MCP 서버 구축


실제 외부 API를 MCP 서버로 통합하여 LLM이 실시간 데이터에 접근할 수 있도록 만드는 프로젝트입니다.
## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [API 서버 소개](#api-서버-소개)
3. [환경 설정](#환경-설정)
4. [서버별 사용 가이드](#서버별-사용-가이드)
5. [통합 실행](#통합-실행)
6. [실전 활용 예시](#실전-활용-예시)
7. [문제 해결](#문제-해결)

---
![](../assets/img/posts/MCP-20251223074727.png)
## 프로젝트 개요

### 무엇을 만들까요?

LLM이 실시간 정보에 접근할 수 있도록 3가지 API 기반 MCP 서버를 구축합니다:

1. **날씨 API 서버** - OpenWeatherMap 
2. **번역 API 서버** - Google Translate (무료)
3. **뉴스 API 서버** - NewsAPI

### 학습 목표

- ✅ 실제 REST API를 MCP 서버로 감싸기
- ✅ API 키 관리 (환경변수)
- ✅ 에러 처리 및 데모 모드 구현
- ✅ 여러 API 서버 동시 운영
- ✅ LLM이 실시간 데이터 활용하기

---

## API 서버 소개

### 1. 날씨 API 서버 (`weather_api_server.py`)

**기능:**
- 현재 날씨 조회
- 5일 날씨 예보

**API:** OpenWeatherMap (무료 플랜)
- 월 1,000회 호출 무료
- API 키 필요
- 발급: https://openweathermap.org/api

**도구:**
- `get_current_weather(city, units)` - 현재 날씨
- `get_weather_forecast(city, units, days)` - 예보

---

### 2. 번역 API 서버 (`translator_server.py`)

**기능:**
- 텍스트 번역
- 언어 자동 감지
- 지원 언어 목록

**API:** Google Translate (무료 버전)
- API 키 불필요
- `googletrans` 라이브러리 사용
- 무제한 무료 (단, 속도 제한 있을 수 있음)

**도구:**
- `translate_text(text, target_lang, source_lang)` - 번역
- `detect_language(text)` - 언어 감지
- `list_languages()` - 지원 언어

---

### 3. 뉴스 API 서버 (`news_api_server.py`)

**기능:**
- 키워드 뉴스 검색
- 국가별 헤드라인

**API:** NewsAPI.org (무료 플랜)
- 하루 100회 호출 무료
- API 키 필요
- 발급: https://newsapi.org/

**도구:**
- `search_news(query, language, sort_by)` - 뉴스 검색
- `get_top_headlines(country, category)` - 헤드라인

---

## 환경 설정

### 1. 필수 패키지 설치

```bash
# 공통
pip install mcp python-dotenv

# 날씨 & 뉴스
pip install requests

# 번역
pip install googletrans==4.0.0rc1
```

### 2. API 키 발급

#### OpenWeatherMap (날씨)
1. https://openweathermap.org/api 방문
2. 회원가입
3. API Keys 메뉴에서 키 복사

#### NewsAPI (뉴스)
1. https://newsapi.org/ 방문
2. 회원가입
3. API 키 복사

#### Google Translate (번역)
- API 키 불필요! 설치만 하면 됨

### 3. 환경변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
# OpenWeatherMap API 키
OPENWEATHER_API_KEY=your_openweather_api_key_here

# NewsAPI 키
NEWS_API_KEY=your_newsapi_key_here
```

**중요:** `.env` 파일은 Git에 커밋하지 마세요!

```bash
# .gitignore에 추가
echo ".env" >> .gitignore
```

---

## 서버별 사용 가이드

### 날씨 API 서버

**파일 위치:** `servers/weather_api_server.py`

**단독 테스트:**
```bash
python servers/weather_api_server.py
```

**도구 사용 예:**
```python
await client.call_mcp_tool(
    "weather",
    "get_current_weather",
    {"city": "Seoul", "units": "metric"}
)
```

**결과 예시:**
```
🌤️ Seoul 현재 날씨

🌡️ 온도: 5.2°C
🤔 체감온도: 2.1°C
📊 최저/최고: 3.0°C / 8.0°C
💧 습도: 65%
🌬️ 풍속: 3.5 m/s
🔽 기압: 1013 hPa
☁️ 날씨: 맑음
```

**데모 모드:**
API 키가 없으면 샘플 데이터로 작동합니다.

---

### 번역 API 서버

**파일 위치:** `servers/translator_server.py`

**단독 테스트:**
```bash
python servers/translator_server.py
```

**도구 사용 예:**
```python
await client.call_mcp_tool(
    "translator",
    "translate_text",
    {
        "text": "Hello, how are you?",
        "target_lang": "ko",
        "source_lang": "en"
    }
)
```

**결과 예시:**
```
🌐 번역 결과

📝 원문 (영어):
Hello, how are you?

✨ 번역문 (한국어):
안녕하세요, 어떻게 지내세요?

💡 신뢰도: 95%
```

**지원 언어:**
- ko (한국어), en (영어), ja (일본어)
- zh-cn (중국어), es (스페인어), fr (프랑스어)
- 등 100개 이상 언어

---

### 뉴스 API 서버

**파일 위치:** `servers/news_api_server.py`

**단독 테스트:**
```bash
python servers/news_api_server.py
```

**도구 사용 예:**
```python
await client.call_mcp_tool(
    "news",
    "search_news",
    {
        "query": "AI",
        "language": "ko",
        "sort_by": "publishedAt"
    }
)
```

**결과 예시:**
```
📰 'AI' 뉴스 검색 결과 (총 10건)

[1] AI 기술 발전으로 새로운 시대 열려
📅 2025-12-23 10:30 | 📰 테크뉴스
📝 최신 인공지능 기술이 다양한 산업에...
🔗 https://example.com/news1

[2] 경제 전망: 2025년 성장 예측
...
```

---

## 통합 실행

### 메인 클라이언트 수정

`hf_client.py`의 `main()` 함수:

```python
async def main():
    print("=" * 70)
    print("🚀 실전 MCP 클라이언트 - 실제 API 통합")
    print("=" * 70)
    
    # 모델 로딩
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    client = HuggingFaceMCPClient(
        model_name=MODEL_NAME,
        use_4bit=True,
        device="auto"
    )
    
    # ✨ API 서버들 연결
    await client.connect_mcp_server("weather", "servers/weather_api_server.py")
    await client.connect_mcp_server("translator", "servers/translator_server.py")
    await client.connect_mcp_server("news", "servers/news_api_server.py")
    
    # 기존 서버들도 함께 사용 가능
    await client.connect_mcp_server("calculator", "servers/calculator_server.py")
    await client.connect_mcp_server("memo", "servers/memo_server.py")
    
    print("\n📋 사용 가능한 도구:")
    for tool in client.available_tools:
        print(f"  - {tool['name']} ({tool['server']}): {tool['description']}")
    
    print("\n" + "=" * 70)
    print("대화를 시작합니다!")
    print("=" * 70)
    
    # 대화 루프
    try:
        while True:
            user_input = input("\n당신: ").strip()
            
            if user_input.lower() in ['exit', 'quit', '종료']:
                break
            
            if not user_input:
                continue
            
            await client.chat(user_input)
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 실행

```bash
python hf_client_fixed.py
```

---

## 실전 활용 예시

### 예시 1: 날씨 기반 여행 추천

```
당신: 서울 날씨 알려줘

🤖: (weather 서버 사용)
🌤️ Seoul 현재 날씨
🌡️ 온도: 5°C, 맑음

당신: 그럼 주말 여행 추천해줘

🤖: 현재 서울 날씨가 맑고 선선하니, 
     남산이나 북한산 등산을 추천드립니다!
```

### 예시 2: 다국어 뉴스 번역

```
당신: AI 관련 영어 뉴스 찾아줘

🤖: (news 서버 사용, language="en")
[1] "AI Revolution in Healthcare"...

당신: 첫 번째 제목 한국어로 번역해줘

🤖: (translator 서버 사용)
✨ 번역문: "의료 분야의 AI 혁명"
```

### 예시 3: 복합 작업

```
당신: 도쿄 날씨 알려주고, 그걸 일본어로 번역해줘

🤖: 
1️⃣ (weather 서버) 도쿄 날씨 조회
2️⃣ (translator 서버) 결과를 일본어로 번역

🌤️ 東京の天気
気温: 8°C、曇り
```

### 예시 4: 여러 서버 연계

```
당신: 부산 날씨 보고 메모에 저장해줘

🤖:
1️⃣ (weather 서버) 부산 날씨 조회
2️⃣ (memo 서버) 메모 저장

✅ 메모 저장 완료!
제목: 부산 날씨 (2025-12-23)
내용: 온도 7°C, 맑음
```

---

## 문제 해결

### 1. API 키 오류

**증상:**
```
❌ 오류: 401 Unauthorized
```

**해결:**
- `.env` 파일에 API 키가 올바르게 입력되었는지 확인
- API 키가 활성화되었는지 확인
- 무료 플랜 한도 초과 여부 확인

### 2. requests 설치 오류

**증상:**
```
⚠️ requests가 설치되지 않았습니다
```

**해결:**
```bash
pip install requests
```

### 3. googletrans 설치 오류

**증상:**
```
AttributeError: 'NoneType' object has no attribute 'group'
```

**해결:**
정확한 버전 설치:
```bash
pip uninstall googletrans
pip install googletrans==4.0.0rc1
```

### 4. 네트워크 타임아웃

**증상:**
```
❌ API 호출 실패: Timeout
```

**해결:**
- 인터넷 연결 확인
- 방화벽 설정 확인
- VPN 사용 시 끄고 재시도

%% ### 5. LLM이 도구를 안 씀

**해결:**
- 7B 이상 모델 사용 (Qwen2.5-7B-Instruct)
- 히스토리 제거 (현재 메시지만 사용)
- 프롬프트에 명시적 지시 추가 %%

---

%% ## 추가 개선 아이디어

### 1. 더 많은 API 추가

- **환율 API**: 실시간 환율 조회
- **주식 API**: 주가 정보
- **지도 API**: 위치 검색, 거리 계산
- **유튜브 API**: 영상 검색

### 2. 에러 처리 강화

```python
# 재시도 로직
max_retries = 3
for attempt in range(max_retries):
    try:
        result = requests.get(url)
        break
    except requests.exceptions.Timeout:
        if attempt == max_retries - 1:
            return {"error": "타임아웃"}
        time.sleep(1)
```

### 3. 캐싱 추가

```python
# 중복 요청 방지
cache = {}

def get_weather_cached(city):
    if city in cache:
        cached_time, data = cache[city]
        if time.time() - cached_time < 600:  # 10분
            return data
    
    data = get_weather_from_api(city)
    cache[city] = (time.time(), data)
    return data
```

### 4. 로깅 추가

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"API 호출: {url}")
logger.error(f"오류 발생: {e}")
```

---

## 다음 단계

1. **API 키 발급 및 설정**
   - OpenWeatherMap
   - NewsAPI

2. **서버 테스트**
   ```bash
   python servers/weather_api_server.py
   python servers/translator_server.py
   python servers/news_api_server.py
   ```

3. **통합 실행**
   ```bash
   python hf_client_fixed.py
   ```

4. **실전 활용**
   - 날씨 조회
   - 번역 요청
   - 뉴스 검색
   - 복합 작업
 %%
---

## 요약

**이제 여러분은:**
- ✅ 실제 REST API를 MCP 서버로 통합할 수 있어요
- ✅ API 키를 안전하게 관리할 수 있어요
- ✅ 여러 API 서버를 동시에 운영할 수 있어요
- ✅ LLM이 실시간 데이터를 활용하게 만들 수 있어요

**다음 프로젝트:**
- RAG 시스템과 MCP 통합
- 데이터베이스 MCP 서버
- 사내 시스템 API 연동

## 다음 단계


**다음: [06-실전-프로젝트-RAG](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_6/#/)**

여러분이 LangGraph로 만들었던 RAG를 MCP 서버로 전환하고, Hugging Face LLM과 연결해봅시다!

