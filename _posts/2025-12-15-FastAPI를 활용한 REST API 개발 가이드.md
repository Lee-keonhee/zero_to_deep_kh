---
layout: post
title: FastAPI를 활용한 REST API 개발 가이드
summary: FastAPI를 활용한 REST API 개발
author: keonhee
date: 2025-12-15 12:00:00 +0900
category: FastAPI
keywords: FastAPI
permalink: /blog/FastAPI_1/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
# FastAPI를 활용한 REST API 개발 가이드

## **1. FastAPI 시작하기**

### 1.1 환경 설정

FastAPI는 Python 기반의 현대적인 웹 프레임워크로, 빠른 성능과 자동 문서화 기능을 제공합니다.

**필수 패키지 설치:**

```bash
pip install fastapi uvicorn
```

### 1.2 프로젝트 디렉토리 구성

```bash
api_project/
    ├── main.py              # API 메인 파일
    └── requirements.txt     # 의존성 패키지 목록
```

### 1.3 기본 API 서버 구축

```python
from fastapi import FastAPI

app = FastAPI()

# 샘플 데이터베이스 (딕셔너리)
items_db = {
    "item1": {"id": "item1", "title": "첫 번째 아이템", "price": 10000},
    "item2": {"id": "item2", "title": "두 번째 아이템", "price": 20000},
}

# 전체 목록 조회 엔드포인트
@app.get("/items")
def read_items():
    return items_db

# 서버 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

**주요 구성 요소:**

- `FastAPI()`: 애플리케이션 인스턴스 생성
- `@app.get()`: GET 요청 처리 데코레이터
- `uvicorn.run()`: ASGI 서버 실행

**자동 문서화 확인:**

- Swagger UI: `http://127.0.0.1:8080/docs`
- ReDoc: `http://127.0.0.1:8080/redoc`

### 1.4 HTTP 메서드와 상태 코드 이해

**주요 HTTP 메서드:**

|메서드|용도|사용 예시|
|---|---|---|
|**GET**|리소스 조회|`/items`, `/items/{id}`|
|**POST**|리소스 생성|`/items`|
|**PUT**|리소스 전체 수정|`/items/{id}`|
|**PATCH**|리소스 부분 수정|`/items/{id}`|
|**DELETE**|리소스 삭제|`/items/{id}`|

**주요 HTTP 상태 코드:**

|코드|의미|사용 상황|
|---|---|---|
|200|OK|요청 성공|
|201|Created|리소스 생성 성공|
|400|Bad Request|잘못된 요청|
|404|Not Found|리소스 없음|
|500|Internal Server Error|서버 오류|

### 1.5 클라이언트 요청 예제

```python
import requests

# API 베이스 URL
API_URL = "http://127.0.0.1:8080/items"

# 전체 아이템 조회
def fetch_all_items():
    res = requests.get(API_URL)
    if res.status_code == 200:
        print("아이템 목록:", res.json())
    else:
        print(f"오류 발생: {res.status_code}")

# 실행
if __name__ == "__main__":
    fetch_all_items()
```

**requests 모듈 주요 기능:**

- `requests.get(url)`: GET 요청 전송
- `response.status_code`: HTTP 상태 코드 확인
- `response.json()`: JSON 응답을 Python 객체로 변환

### 1.6 경로 파라미터 활용

**서버 코드:**

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

items_db = {
    "item1": {"id": "item1", "title": "첫 번째 아이템", "price": 10000},
    "item2": {"id": "item2", "title": "두 번째 아이템", "price": 20000},
}

# 특정 아이템 조회
@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id in items_db:
        return JSONResponse(content=items_db[item_id], status_code=200)
    return JSONResponse(content={"message": "아이템을 찾을 수 없습니다"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

**클라이언트 코드:**

```python
import requests

API_URL = "http://127.0.0.1:8080/items"

# 특정 아이템 조회
def fetch_item(item_id):
    res = requests.get(f"{API_URL}/{item_id}")
    if res.status_code == 200:
        print(f"아이템 {item_id}:", res.json())
    else:
        print(f"조회 실패: {res.status_code}, {res.json()}")

if __name__ == "__main__":
    fetch_item('item1')
    fetch_item('item999')  # 존재하지 않는 아이템
```

## 2. 요청 데이터 처리 (Pydantic 활용)

### 2.1 데이터 모델 정의 및 POST 요청

```python
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

items_db = {
    "item1": {"id": "item1", "title": "첫 번째 아이템", "price": 10000},
}

# Pydantic 모델 정의
class Item(BaseModel):
    id: str
    title: str
    price: int

# 아이템 생성 API
@app.post("/items")
def create_item(item: Item):
    if item.id in items_db:
        return JSONResponse(
            content={"message": "이미 존재하는 아이템입니다"}, 
            status_code=400
        )
    items_db[item.id] = item.model_dump()
    return JSONResponse(content=items_db[item.id], status_code=201)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

**Pydantic의 장점:**

- 자동 데이터 검증
- 타입 힌팅 지원
- 자동 문서화
- JSON 스키마 생성

### 2.2 선택적 필드를 포함한 PUT 요청

```python
from pydantic import BaseModel, Field
from typing import Optional

# 수정용 모델 (모든 필드 선택적)
class ItemUpdate(BaseModel):
    title: Optional[str] = Field(
        default=None, 
        max_length=50,
        description="아이템 제목"
    )
    price: Optional[int] = Field(
        default=None, 
        ge=0, 
        le=1000000,
        description="아이템 가격 (0~1,000,000)"
    )

# 아이템 수정 API
@app.put("/items/{item_id}")
def update_item(item_id: str, item: ItemUpdate):
    if item_id not in items_db:
        return JSONResponse(
            content={"message": "아이템을 찾을 수 없습니다"}, 
            status_code=404
        )
    
    update_data = item.model_dump(exclude_none=True)    # exclude_none을 사용하면 None 값은 저절로 제외하고 변경함.
    items_db[item_id].update(update_data)
    
    return JSONResponse(content=items_db[item_id], status_code=200)
```

**Field 파라미터:**

- `default`: 기본값 설정
- `max_length`: 문자열 최대 길이
- `ge`, `le`: 숫자의 최소/최대값 (이상, 이하)
- `gt`, `lt`: 숫자의 최소/최대값 (초과, 미만)
- `description`: 필드 설명
- `examples`: 예시 값

### 2.3 완성된 서버 코드

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI(title="상품 관리 API")

# 데이터 저장소
items_db = {
    "item1": {"id": "item1", "title": "첫 번째 아이템", "price": 10000},
    "item2": {"id": "item2", "title": "두 번째 아이템", "price": 20000},
}

# 전체 조회
@app.get("/items")
def read_items():
    return items_db

# 개별 조회
@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id in items_db:
        return JSONResponse(content=items_db[item_id], status_code=200)
    return JSONResponse(content={"message": "아이템을 찾을 수 없습니다"}, status_code=404)

# 생성용 모델
class Item(BaseModel):
    id: str
    title: str
    price: int

# 아이템 생성
@app.post("/items")
def create_item(item: Item):
    if item.id in items_db:
        return JSONResponse(
            content={"message": "이미 존재하는 아이템입니다"}, 
            status_code=400
        )
    items_db[item.id] = item.model_dump()
    return JSONResponse(content=items_db[item.id], status_code=201)

# 수정용 모델
class ItemUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=50)
    price: Optional[int] = Field(default=None, ge=0, le=1000000)

# 아이템 수정
@app.put("/items/{item_id}")
def update_item(item_id: str, item: ItemUpdate):
    if item_id not in items_db:
        return JSONResponse(
            content={"message": "아이템을 찾을 수 없습니다"}, 
            status_code=404
        )
    
    update_data = item.model_dump(exclude_none=True)
    items_db[item_id].update(update_data)
    
    return JSONResponse(content=items_db[item_id], status_code=200)

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

### 2.4 완성된 클라이언트 코드

```python
import requests

API_URL = "http://127.0.0.1:8080/items"

# 전체 조회
def fetch_all_items():
    res = requests.get(API_URL)
    if res.status_code == 200:
        print("전체 아이템:", res.json())
    else:
        print(f"조회 실패: {res.status_code}")

# 개별 조회
def fetch_item(item_id):
    res = requests.get(f"{API_URL}/{item_id}")
    if res.status_code == 200:
        print(f"아이템 {item_id}:", res.json())
    else:
        print(f"조회 실패: {res.status_code}")

# 아이템 생성
def create_item(item_id, title, price):
    data = {"id": item_id, "title": title, "price": price}
    res = requests.post(API_URL, json=data)
    if res.status_code == 201:
        print("생성 완료:", res.json())
    else:
        print(f"생성 실패: {res.status_code}, {res.json()}")

# 아이템 수정
def update_item(item_id, title=None, price=None):
    data = {}
    if title:
        data["title"] = title
    if price is not None:
        data["price"] = price
    
    res = requests.put(f"{API_URL}/{item_id}", json=data)
    if res.status_code == 200:
        print(f"수정 완료:", res.json())
    else:
        print(f"수정 실패: {res.status_code}, {res.json()}")

# 테스트 실행
if __name__ == "__main__":
    fetch_all_items()
    fetch_item('item1')
    create_item('item3', '세 번째 아이템', 30000)
    update_item('item1', price=15000)
    update_item('item2', title='수정된 제목')
```

### 2.5 API 문서화 개선

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI(
    title="상품 관리 API",
    description="RESTful API를 통한 상품 CRUD 시스템",
    version="1.0.0"
)

items_db = {
    "item1": {"id": "item1", "title": "첫 번째 아이템", "price": 10000},
}

@app.get(
    "/items", 
    summary="전체 아이템 목록 조회",
    response_description="등록된 모든 아이템 정보"
)
def read_items():
    """
    모든 아이템의 정보를 조회합니다.
    """
    return items_db

@app.get(
    "/items/{item_id}",
    summary="특정 아이템 조회",
    response_description="요청한 아이템의 상세 정보"
)
def read_item(item_id: str):
    """
    ID를 통해 특정 아이템의 정보를 조회합니다.
    
    - **item_id**: 조회할 아이템의 고유 ID
    """
    if item_id in items_db:
        return JSONResponse(content=items_db[item_id], status_code=200)
    return JSONResponse(content={"message": "아이템을 찾을 수 없습니다"}, status_code=404)

class Item(BaseModel):
    id: str = Field(..., description="아이템 고유 ID", examples=["item1"])
    title: str = Field(..., description="아이템 제목", examples=["노트북"])
    price: int = Field(..., ge=0, description="가격 (원)", examples=[100000])
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "item3",
                    "title": "무선 마우스",
                    "price": 25000
                }
            ]
        }
    }

@app.post(
    "/items",
    summary="새 아이템 등록",
    response_description="생성된 아이템 정보",
    status_code=201
)
def create_item(item: Item):
    """
    새로운 아이템을 등록합니다.
    
    - **id**: 아이템 고유 ID (중복 불가)
    - **title**: 아이템 제목
    - **price**: 아이템 가격
    """
    if item.id in items_db:
        return JSONResponse(
            content={"message": "이미 존재하는 아이템입니다"}, 
            status_code=400
        )
    items_db[item.id] = item.model_dump()
    return JSONResponse(content=items_db[item.id], status_code=201)

class ItemUpdate(BaseModel):
    title: Optional[str] = Field(
        default=None, 
        max_length=50,
        description="변경할 아이템 제목",
        examples=["수정된 제목"]
    )
    price: Optional[int] = Field(
        default=None, 
        ge=0, 
        le=1000000,
        description="변경할 가격 (0~1,000,000원)",
        examples=[50000]
    )

@app.put(
    "/items/{item_id}",
    summary="아이템 정보 수정",
    response_description="수정된 아이템 정보"
)
def update_item(item_id: str, item: ItemUpdate):
    """
    기존 아이템의 정보를 수정합니다.
    
    - **item_id**: 수정할 아이템의 ID
    - **title**: 변경할 제목 (선택사항)
    - **price**: 변경할 가격 (선택사항)
    """
    if item_id not in items_db:
        return JSONResponse(
            content={"message": "아이템을 찾을 수 없습니다"}, 
            status_code=404
        )
    
    update_data = item.model_dump(exclude_none=True)
    items_db[item_id].update(update_data)
    
    return JSONResponse(content=items_db[item_id], status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

**문서화 개선 포인트:**

- `summary`: 엔드포인트 요약 설명
- `response_description`: 응답 데이터 설명
- 독스트링(""")을 통한 상세 설명
- `Field()`의 `description`과 `examples`로 필드별 상세 정보 제공
- `model_config`로 전체 모델 예시 제공

**문서 확인:** `http://127.0.0.1:8080/docs`에서 향상된 문서 확인 가능