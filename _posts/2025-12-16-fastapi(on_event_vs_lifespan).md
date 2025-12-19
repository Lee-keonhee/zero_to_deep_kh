---
layout: post
title: FastAPI의 on_event_vs_lifespan
summary: FastAPI를 활용한 REST API 개발
author: keonhee
date: 2025-12-16 12:00:00 +0900
category: FastAPI
keywords: FastAPI
permalink: /blog/FastAPI_2/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---

# FastAPI Lifespan Event 완벽 가이드

## **1. Lifespan Event 개요**

### 1.1 Lifespan Event란?

FastAPI 애플리케이션을 개발하다 보면, 애플리케이션 시작 시와 종료 시 특정 작업이 필요할 때가 있습니다.

**주요 사용 사례:**

- 데이터베이스 연결: 
	애플리케이션이 시작될 때 데이터베이스 연결 풀을 초기화하고, 종료될 때 연결을 해제합니다.
- 스케줄러 작업 초기화: 
	시작 시 작업 스케줄러(APScheduler 등)를 설정하고 실행하며, 종료 시 안전하게 스케줄러를 중지합니다.
- 머신러닝 모델 로드: 
	대규모 머신러닝 모델을 애플리케이션 시작 시 메모리에 로드하여, 요청 간에 공유될 수 있도록 설정합니다.
- 캐시 시스템 설정: 
	시작 시 Redis와 같은 캐시를 초기화하고 종료 시 연결을 해제합니다.
- 메시지 큐 설정: 
	RabbitMQ, Kafka와 같은 메시지 큐를 시작 시 연결하고 종료 시 이를 안전하게 정리합니다.
- 외부 API 인증 토큰 갱신: 
	시작 시 외부 API 토큰을 가져오고 종료 시 안전하게 정리합니다.

이처럼, 애플리케이션의 수명 주기(lifecycle) 동안 리소스를 초기화하거나 정리해야 하는 경우에 FastAPI는 Lifespan을 사용합니다.

### 1.2 애플리케이션의 수명 주기

애플리케이션의 수명 주기(lifecycle)는 애플리케이션이 시작되어 실행되고 종료되기까지의 과정을 말합니다.

**수명 주기 단계:**

- **애플리케이션 시작**: 서버가 시작되며 필요한 리소스를 초기화 (데이터베이스 연결 생성, 스케줄러 시작, 설정 파일 로드 등)
- **애플리케이션 실행**: 사용자의 요청을 처리하며 정상적으로 동작
- **애플리케이션 종료**: 서버가 종료되며 리소스를 해제 (데이터베이스 연결 닫기, 메모리 정리, 로그 저장 등)

**FastAPI 애플리케이션 수명 주기 예시:**

- 시작: `uvicorn main:app` 실행 → 서버 시작
- 실행: 클라이언트 요청 처리(HTTP 요청/응답)
- 종료: 서버가 중지되거나 충돌 → 리소스 정리

### 1.3 컨텍스트 관리

컨텍스트는 애플리케이션이 특정 작업을 수행하기 위해 필요한 상태와 환경입니다.

**예시:**

애플리케이션이 데이터베이스와 연결을 사용한다고 가정할 때:

- 시작 시: 데이터베이스 연결을 초기화
- 실행 중: 연결을 사용해 쿼리 처리
- 종료 시: 연결을 닫아 리소스를 해제

컨텍스트 관리란, 이러한 초기화와 정리 작업을 자동으로 처리하여 리소스 누수를 방지하고 코드의 가독성을 높이는 것을 의미합니다.

---

## **2. 기존 방식: @app.on_event()**

### 2.1 @app.on_event() 사용 방법

과거에는 `@app.on_event("startup")` 및 `@app.on_event("shutdown")` 데코레이터를 사용하여 애플리케이션의 시작과 종료 작업을 관리했습니다.

```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup():
    # 애플리케이션이 시작되기 전 실행
    print("데이터베이스 연결 설정 중...")

@app.on_event("shutdown")
async def shutdown():
    # 애플리케이션이 종료되기 전 실행
    print("리소스 정리 중...")
```

**동작 방식:**

- 애플리케이션이 시작될 때 필요한 작업은 별도의 함수로 정의한 후, 해당 함수에 `@app.on_event("startup")` 데코레이터를 적용하여 등록
- 종료 시 실행되어야 할 작업은 `@app.on_event("shutdown")` 데코레이터를 사용해 동일한 방식으로 정의

### 2.2 여러 이벤트 핸들러 등록

`@app.on_event()`는 여러 함수에 적용할 수 있습니다.

```python
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_task_1():
    print("Startup Task 1: Initialize database connection.")

@app.on_event("startup")
async def startup_task_2():
    print("Startup Task 2: Initialize background scheduler.")

@app.on_event("shutdown")
async def shutdown_task_1():
    print("Shutdown Task 1: Close database connection.")

@app.on_event("shutdown")
async def shutdown_task_2():
    print("Shutdown Task 2: Stop background scheduler.")
```

**실행 결과:**

```bash
# FastAPI 애플리케이션 실행 시
Startup Task 1: Initialize database connection.
Startup Task 2: Initialize background scheduler.
INFO:     Application startup complete.

...

# FastAPI 애플리케이션 종료 시:
Shutdown Task 1: Close database connection.
Shutdown Task 2: Stop background scheduler.
INFO:     Application shutdown complete.
```

같은 이벤트(startup 또는 shutdown)에 대해 여러 개의 함수를 등록할 수 있으며, 이 함수들은 등록된 순서대로 실행됩니다.

### 2.3 @app.on_event()의 한계점

**1. 시작과 종료 로직 분리**

시작과 종료 작업이 별도의 함수로 분리되어 있어, 이들 간에 데이터를 공유하려면 전역 변수나 기타 우회적인 방법을 사용해야 합니다.

```python
# 전역 변수 사용 필요
db_connection = None

@app.on_event("startup")
async def startup():
    global db_connection
    db_connection = await create_db_connection()

@app.on_event("shutdown")
async def shutdown():
    global db_connection
    if db_connection:
        await db_connection.close()
```

**2. 리소스 관리 복잡성**

- 초기화된 리소스를 종료 시 정확히 해제해야 하는지 추적하기 어려움
- 예외 발생 시 리소스 정리가 보장되지 않음
- 여러 리소스를 관리할 때 코드가 복잡해짐

**3. Deprecated 상태**

FastAPI는 현재 `@app.on_event()` 사용을 권장하지 않으며, Lifespan Event 사용을 권장합니다.

---

## **3. 새로운 방식: Lifespan Event**

### 3.1 Lifespan 기본 사용법

Lifespan event은 애플리케이션 시작과 종료 시 필요한 로직을 각각 개별적으로 등록하는 대신, 하나의 함수에서 관리합니다.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션이 시작되기 전 실행
    print("데이터베이스 연결 설정 중...")
    
    yield
    
    # 애플리케이션이 종료되기 전 실행
    print("리소스 정리 중...")

app = FastAPI(lifespan=lifespan)
```

**핵심 개념:**

- `yield` 키워드를 기준으로, 이전에는 애플리케이션 시작 시 실행되는 작업, 이후에는 애플리케이션 종료 시 실행되는 작업을 정의
- Python의 `contextlib` 모듈에서 제공하는 `@asynccontextmanager` 데코레이터를 사용
- FastAPI 인스턴스를 생성할 때 `lifespan` 매개변수로 등록

### 3.2 여러 작업 관리

Lifespan event는 애플리케이션당 한 번만 정의할 수 있지만, 함수 내부에서 여러 초기화 작업이나 정리 작업을 포함할 수 있습니다.

작업을 분리해서 관리하려면 lifespan 내부에 여러 함수를 호출하면 됩니다.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

async def initialize_database():
    print("Database initialized.")

async def initialize_scheduler():
    print("Scheduler initialized.")

async def clean_up_database():
    print("Database connection closed.")

async def clean_up_scheduler():
    print("Scheduler stopped.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    await initialize_database()
    await initialize_scheduler()
    print("All startup tasks completed.")
    
    yield
    
    # Shutdown tasks
    await clean_up_database()
    await clean_up_scheduler()
    print("All shutdown tasks completed.")

app = FastAPI(lifespan=lifespan)
```

e**실행 결과:**

```bash
# FastAPI 애플리케이션 실행 시
Database initialized.
Scheduler initialized.
All startup tasks completed.
INFO:     Application startup complete.

...

# FastAPI 애플리케이션 종료 시:
Database connection closed.
Scheduler stopped.
All shutdown tasks completed.
INFO:     Application shutdown complete.
```

### 3.3 app.state를 통한 데이터 공유

Lifespan의 가장 큰 장점 중 하나는 `app.state`를 통해 초기화된 리소스를 애플리케이션 전체에서 공유할 수 있다는 것입니다.

```python
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

async def create_db_connection():
    # 실제로는 데이터베이스 연결 생성
    return {"connection": "database_connected"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 데이터베이스 연결 풀 생성 및 저장
    app.state.db = await create_db_connection()
    print(f"데이터베이스 연결됨: {app.state.db}")
    
    yield
    
    # 종료 시 연결 풀 닫기
    print("데이터베이스 연결 해제 중...")
    app.state.db = None

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root(request: Request):
    # 애플리케이션 상태에서 데이터베이스 연결 가져오기
    db = request.app.state.db
    return {"message": "Database connected", "db": str(db)}
```

**주요 특징:**

- 초기화 단계에서 `app.state.db`에 데이터베이스 연결을 저장
- 이후 요청 처리에서 `request.app.state.db`를 통해 직접 접근 가능
- 전역 변수 없이 상태 유지 가능
- 애플리케이션 종료 시 데이터베이스 연결을 안전하게 해제

### 3.4 yield를 통한 상태 유지

yield를 통해 시작 시 생성한 객체가 종료 시점까지 유지됩니다.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    resource = {"connection": "connected"}
    print(f"리소스 생성: {resource}")
    
    yield resource  # yield를 통해 생성한 객체를 유지
    
    resource["connection"] = "disconnected"
    print(f"리소스 정리: {resource}")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get_resource():
    return {"status": "running"}
```

### 3.5 예외 처리

Lifespan에서 예외 처리를 통해 안전하게 리소스를 관리할 수 있습니다.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 리소스 초기화
        print("리소스 초기화 시작...")
        db = await create_db_connection()
        app.state.db = db
        print("리소스 초기화 완료")
        
        yield
        
    except Exception as e:
        print(f"초기화 중 오류 발생: {e}")
    finally:
        # 예외 발생 여부와 관계없이 정리
        print("리소스 정리 중...")
        if hasattr(app.state, "db"):
            await app.state.db.close()
        print("리소스 정리 완료")

app = FastAPI(lifespan=lifespan)
```

---

## **4. Lifespan과 @app.on_event() 비교**

### 4.1 비교표

|구분|@app.on_event()|Lifespan Event|
|---|---|---|
|**구조**|분리된 함수|단일 컨텍스트|
|**데이터 공유**|전역 변수 필요|app.state 활용|
|**상태**|Deprecated|권장 방식|
|**ASGI 호환**|제한적|완전 호환|
|**예외 처리**|복잡|간단 (try-finally)|
|**정의 개수**|여러 개 가능|한 개만 가능|
|**리소스 관리**|수동|구조적|

### 4.2 왜 Lifespan을 사용해야 하나?

**1. 구조적 리소스 관리**

시작과 종료 작업이 서로 연결되는 경우가 많습니다. Lifespan은 yield를 기준으로 초기화한 데이터를 함수 내부의 상태로 유지할 수 있어, 리소스 관리가 명확합니다.

**2. ASGI 프로토콜 준수**

FastAPI는 ASGI(Asynchronous Server Gateway Interface) 기반으로 동작하며, Lifespan은 ASGI의 수명 주기 관리 프로토콜을 준수하여 비동기 서버 환경과 자연스럽게 연동됩니다.

ASGI 서버는 애플리케이션 시작 시 `lifespan.startup` 메시지를 보내고, 종료 시 `lifespan.shutdown` 메시지를 통해 리소스를 정리합니다.

**3. 비동기 환경 최적화**

Lifespan은 Python의 `asynccontextmanager`를 기반으로 하여, yield를 활용해 비동기 리소스를 구조적으로 관리할 수 있습니다. 모든 비동기 작업이 하나의 컨텍스트에서 실행되므로 예외 발생 시 처리가 용이합니다.

**4. Starlette 기반 구현**

FastAPI의 lifespan의 실질적인 구현은 Starlette입니다. (FastAPI는 Starlette를 기반으로 만들어진 프레임워크입니다.)

FastAPI 클래스는 Starlette의 lifespan 매개변수를 받아 애플리케이션 생애 주기를 관리합니다.

---

## **5. 실전 예제**

### 5.1 데이터베이스 연결 관리

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import asyncpg

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 데이터베이스 연결 풀 생성
    print("데이터베이스 연결 풀 생성 중...")
    app.state.pool = await asyncpg.create_pool(
        host="localhost",
        database="mydb",
        user="user",
        password="password",
        min_size=10,
        max_size=20
    )
    print("데이터베이스 연결 풀 생성 완료")
    
    yield
    
    # 연결 풀 닫기
    print("데이터베이스 연결 풀 종료 중...")
    await app.state.pool.close()
    print("데이터베이스 연결 풀 종료 완료")

app = FastAPI(lifespan=lifespan)

@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    async with request.app.state.pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        return dict(user) if user else {"error": "User not found"}
```

### 5.2 머신러닝 모델 로딩

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import pickle

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로딩
    print("머신러닝 모델 로딩 중...")
    with open("model.pkl", "rb") as f:
        app.state.ml_model = pickle.load(f)
    print("모델 로딩 완료")
    
    yield
    
    # 모델 언로드
    print("모델 메모리 해제 중...")
    app.state.ml_model = None
    print("모델 메모리 해제 완료")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(data: dict, request: Request):
    model = request.app.state.ml_model
    prediction = model.predict([data["features"]])
    return {"prediction": prediction.tolist()}
```

### 5.3 스케줄러 설정

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

async def scheduled_task():
    print("스케줄된 작업 실행 중...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 스케줄러 시작
    print("스케줄러 시작 중...")
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_task, "interval", minutes=5)
    scheduler.start()
    app.state.scheduler = scheduler
    print("스케줄러 시작 완료")
    
    yield
    
    # 스케줄러 종료
    print("스케줄러 종료 중...")
    app.state.scheduler.shutdown()
    print("스케줄러 종료 완료")

app = FastAPI(lifespan=lifespan)
```

### 5.4 Redis 캐시 연결

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import aioredis

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Redis 연결
    print("Redis 연결 중...")
    app.state.redis = await aioredis.create_redis_pool("redis://localhost")
    print("Redis 연결 완료")
    
    yield
    
    # Redis 연결 종료
    print("Redis 연결 종료 중...")
    app.state.redis.close()
    await app.state.redis.wait_closed()
    print("Redis 연결 종료 완료")

app = FastAPI(lifespan=lifespan)

@app.get("/cache/{key}")
async def get_cache(key: str, request: Request):
    value = await request.app.state.redis.get(key)
    return {"key": key, "value": value.decode() if value else None}

@app.post("/cache/{key}")
async def set_cache(key: str, value: str, request: Request):
    await request.app.state.redis.set(key, value)
    return {"message": "Cached successfully"}
```

### 5.5 복합 리소스 관리

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import asyncpg
import aioredis
import pickle

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 여러 리소스 초기화
    print("=== 리소스 초기화 시작 ===")
    
    # 데이터베이스
    print("1. 데이터베이스 연결 중...")
    app.state.pool = await asyncpg.create_pool(
        host="localhost", database="mydb"
    )
    
    # Redis
    print("2. Redis 연결 중...")
    app.state.redis = await aioredis.create_redis_pool("redis://localhost")
    
    # ML 모델
    print("3. ML 모델 로딩 중...")
    with open("model.pkl", "rb") as f:
        app.state.model = pickle.load(f)
    
    print("=== 리소스 초기화 완료 ===")
    
    yield
    
    # 여러 리소스 정리
    print("=== 리소스 정리 시작 ===")
    
    print("1. 데이터베이스 연결 종료 중...")
    await app.state.pool.close()
    
    print("2. Redis 연결 종료 중...")
    app.state.redis.close()
    await app.state.redis.wait_closed()
    
    print("3. ML 모델 메모리 해제 중...")
    app.state.model = None
    
    print("=== 리소스 정리 완료 ===")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check(request: Request):
    return {
        "database": "connected" if request.app.state.pool else "disconnected",
        "redis": "connected" if request.app.state.redis else "disconnected",
        "model": "loaded" if request.app.state.model else "not loaded"
    }
```

---

## **6. 주의사항 및 팁**

### 6.1 중요 참고사항

**lifespan 방식과 이벤트 핸들러 방식은 함께 사용할 수 없습니다.**

FastAPI() 선언 시 lifespan 매개변수를 사용하면 startup 및 shutdown 이벤트 핸들러는 더 이상 호출되지 않습니다. 둘 중 하나만 선택해야 합니다.

```python
# ❌ 잘못된 사용 - 함께 사용 불가
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan startup")
    yield
    print("Lifespan shutdown")

app = FastAPI(lifespan=lifespan)

@app.on_event("startup")  # 이 핸들러는 호출되지 않음!
async def startup():
    print("This won't run!")
```

### 6.2 베스트 프랙티스

**1. 의존성 순서 고려**

리소스 간 의존성이 있다면 초기화 순서를 고려해야 합니다.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 순서 중요: 데이터베이스 먼저, 그 다음 캐시
    app.state.db = await create_db()
    app.state.cache = await create_cache(app.state.db)
    
    yield
    
    # 역순으로 정리
    await app.state.cache.close()
    await app.state.db.close()
```

**2. 에러 로깅**

초기화 실패 시 명확한 로그 남기기

```python
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.db = await create_db()
    except Exception as e:
        logging.error(f"DB 초기화 실패: {e}")
        raise
    
    yield
    
    try:
        await app.state.db.close()
    except Exception as e:
        logging.error(f"DB 종료 실패: {e}")
```

**3. 헬스 체크 엔드포인트**

리소스 상태를 확인할 수 있는 엔드포인트 제공

```python
@app.get("/health")
async def health_check(request: Request):
    checks = {
        "database": hasattr(request.app.state, "db"),
        "cache": hasattr(request.app.state, "redis"),
        "model": hasattr(request.app.state, "model")
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
    )
```

### 6.3 테스팅

Lifespan이 있는 FastAPI 앱을 테스트할 때는 TestClient를 사용합니다.

```python
from fastapi.testclient import TestClient

def test_app():
    with TestClient(app) as client:
        # Lifespan의 startup이 자동 실행됨
        response = client.get("/")
        assert response.status_code == 200
        # TestClient 종료 시 Lifespan의 shutdown이 자동 실행됨
```

---

## **7. 정리**

### 7.1 핵심 요약

|항목|내용|
|---|---|
|**Lifespan이란?**|애플리케이션 수명 주기 동안 리소스를 관리하는 FastAPI 기능|
|**사용 이유**|DB 연결, ML 모델 로딩, 스케줄러 등 초기화/정리 작업 필요|
|**기존 방식**|@app.on_event() (Deprecated)|
|**새 방식**|Lifespan Event (권장)|
|**핵심 장점**|app.state 활용, 구조적 관리, ASGI 호환|
|**주의사항**|lifespan과 on_event() 동시 사용 불가|

### 7.2 마이그레이션 가이드

기존 `@app.on_event()` 코드를 Lifespan으로 변환하는 방법:

**변환 전:**

```python
db_connection = None

@app.on_event("startup")
async def startup():
    global db_connection
    db_connection = await create_db_connection()

@app.on_event("shutdown")
async def shutdown():
    global db_connection
    await db_connection.close()
```

**변환 후:**

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup 로직
    app.state.db = await create_db_connection()
    
    yield
    
    # shutdown 로직
    await app.state.db.close()

app = FastAPI(lifespan=lifespan)
```
