---
layout: post
title: MCP에서 Pydantic BaseModel 사용 시 JSON Schema $ref 해소 필수
summary: FastMCP와 Pydantic BaseModel을 사용할 때 발생하는 $ref 참조 문제와 LLM 통합을 위한 해결 방법
author: keonhee
date: 2025-12-28 23:30:00 +0900
category: MCP
keywords: MCP, Pydantic, BaseModel, JSON Schema, $ref, FastMCP, LLM
permalink: /blog/mcp-pydantic-basemodel-ref-resolution/
usemathjax: false
thumbnail: /assets/img/posts/mcp.png
imageNameKey: mcp
---

# MCP에서 Pydantic BaseModel 사용 시 주의사항

## 목차
1. [문제 상황](#1-문제-상황)
2. [원인 분석](#2-원인-분석)
3. [해결 방법](#3-해결-방법)
4. [실전 구현](#4-실전-구현)
5. [대안 및 베스트 프랙티스](#5-대안-및-베스트-프랙티스)

---

## 1. 문제 상황

### MCP 도구 정의 시 BaseModel 사용
```python
# nara_api.py
from pydantic import BaseModel, Field
from fastmcp import FastMCP

app = FastMCP("nara-api")

class BidSearchParams(BaseModel):
    page: str = Field(default="1", description="페이지 번호")
    rows: str = Field(default="1", description="한 페이지당 조회 건수")
    inqryBgnDt: str = Field(..., description="조회 시작일시", pattern=r"^\d{12}$")
    inqryEndDt: str = Field(..., description="조회 종료일시", pattern=r"^\d{12}$")
    bidNtceNo: str = Field(default="", description="입찰 공고번호")

@app.tool()
def get_contstruction_bids(params: BidSearchParams):
    """나라 장터의 공사 입찰 정보를 조회합니다."""
    # ...
```

### 발생하는 JSON Schema 구조

FastMCP가 생성하는 `tool.inputSchema`:
```json
{
  "$defs": {
    "BidSearchParams": {
      "properties": {
        "page": {"type": "string", "default": "1", "description": "페이지 번호"},
        "rows": {"type": "string", "default": "1", "description": "한 페이지당 조회 건수"},
        "inqryBgnDt": {"type": "string", "pattern": "^\\d{12}$", "description": "조회 시작일시"},
        "inqryEndDt": {"type": "string", "pattern": "^\\d{12}$", "description": "조회 종료일시"},
        "bidNtceNo": {"type": "string", "default": "", "description": "입찰 공고번호"}
      },
      "required": ["page", "rows", "inqryBgnDt", "inqryEndDt"],
      "type": "object"
    }
  },
  "properties": {
    "params": {"$ref": "#/$defs/BidSearchParams"}  // ← 문제!
  },
  "required": ["params"],
  "type": "object"
}
```

### LLM이 받는 불완전한 정보
```python
# client.py에서 도구 정보 수집
props = tool['input_schema'].get('properties', {})
# → {"params": {"$ref": "#/$defs/BidSearchParams"}}
```

**결과:** LLM은 `BidSearchParams`가 어떤 필드를 가지는지 전혀 알 수 없음 ❌

---

## 2. 원인 분석

### JSON Schema의 `$ref` 참조 메커니즘

JSON Schema는 중복을 피하기 위해 `$defs`에 정의를 모아두고 `$ref`로 참조하는 방식을 사용합니다.

**왜 이렇게 설계되었나?**
```python
# 같은 스키마를 여러 곳에서 사용하는 경우
class BidSearchParams(BaseModel):
    # ... fields

@app.tool()
def get_construction_bids(params: BidSearchParams):
    pass

@app.tool()
def get_service_bids(params: BidSearchParams):  # 같은 스키마 재사용
    pass
```

`$ref`를 사용하면:
- ✅ 중복 정의 방지 (DRY 원칙)
- ✅ 일관성 유지
- ✅ 스키마 크기 감소

하지만 **LLM에게 직접 전달하면:**
- ❌ 참조만 있고 실제 필드 정보 없음
- ❌ LLM이 올바른 파라미터 생성 불가능

### Pydantic의 동작 방식
```python
# Pydantic이 자동 생성하는 JSON Schema
BidSearchParams.model_json_schema()

# 출력:
{
  "$defs": {"BidSearchParams": {...}},
  "properties": {"params": {"$ref": "#/$defs/BidSearchParams"}}
}
```

FastMCP, OpenAPI, Pydantic 모두 이 방식을 표준으로 사용합니다.

---

## 3. 해결 방법

### 3-1. `$ref` 해소 함수 구현 (권장 ✅)

**핵심 아이디어:** `$ref`를 만나면 `$defs`에서 실제 정의를 가져와 치환
```python
# client.py
def _resolve_schema_refs(self, schema: Any, definitions: Dict) -> Any:
    """JSON Schema의 $ref를 실제 스키마로 치환"""
    if isinstance(schema, dict):
        # $ref가 있으면 해당 정의로 교체
        if "$ref" in schema:
            # "#/$defs/BidSearchParams" → "BidSearchParams" 추출
            ref_path = schema["$ref"].split("/")[-1]
            if ref_path in definitions:
                # 재귀적으로 해소 (nested refs 처리)
                return self._resolve_schema_refs(definitions[ref_path], definitions)
        
        # 모든 키-값을 재귀적으로 처리
        return {k: self._resolve_schema_refs(v, definitions) 
                for k, v in schema.items()}
    
    elif isinstance(schema, list):
        return [self._resolve_schema_refs(item, definitions) 
                for item in schema]
    
    return schema
```

### 3-2. MCP 도구 수집 시 적용
```python
async def _connect_mcp_server(self, server_name: str, command: str, args: List):
    """MCP 서버 연결 및 도구 수집"""
    server_params = StdioServerParameters(command=command, args=args)
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()

            for tool in tools_response.tools:
                # 1. $defs 추출
                definitions = tool.inputSchema.get("$defs", {})
                
                # 2. properties에서 $ref 해소
                resolved_properties = self._resolve_schema_refs(
                    tool.inputSchema.get("properties", {}),
                    definitions
                )
                
                # 3. 해소된 스키마 저장
                self.available_tools.append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "properties": resolved_properties,
                        "required": tool.inputSchema.get("required", [])
                    },
                })
                
                logger.info(f"✅ {tool.name} - Schema resolved")
```

### 3-3. 결과 비교

**Before (해소 전):**
```
- Parameters: {"params": {"$ref": "#/$defs/BidSearchParams"}}
```

**After (해소 후):**
```
- Parameters: {
    "params": {
      "type": "object",
      "properties": {
        "page": {"type": "string", "default": "1", "description": "페이지 번호"},
        "rows": {"type": "string", "default": "1", "description": "한 페이지당 조회 건수"},
        "inqryBgnDt": {"type": "string", "pattern": "^\\d{12}$", "description": "조회 시작일시"},
        "inqryEndDt": {"type": "string", "pattern": "^\\d{12}$", "description": "조회 종료일시"},
        "bidNtceNo": {"type": "string", "default": "", "description": "입찰 공고번호"}
      },
      "required": ["page", "rows", "inqryBgnDt", "inqryEndDt"]
    }
  }
```

---

## 4. 실전 구현

### 전체 클라이언트 코드 예시
```python
class HFMCPClient(BaseLLMClient):
    def __init__(self, model_name: str, mcpServers: Dict):
        self.available_tools = []
        # MCP 서버 연결...
        
    def _resolve_schema_refs(self, schema: Any, definitions: Dict) -> Any:
        """$ref 해소"""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"].split("/")[-1]
                if ref_path in definitions:
                    return self._resolve_schema_refs(definitions[ref_path], definitions)
            return {k: self._resolve_schema_refs(v, definitions) 
                    for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._resolve_schema_refs(item, definitions) 
                    for item in schema]
        return schema
    
    async def _connect_mcp_server(self, server_name: str, command: str, args: List):
        """MCP 도구 등록 (스키마 해소 포함)"""
        server_params = StdioServerParameters(command=command, args=args)
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                for tool in tools_response.tools:
                    definitions = tool.inputSchema.get("$defs", {})
                    resolved_properties = self._resolve_schema_refs(
                        tool.inputSchema.get("properties", {}),
                        definitions
                    )
                    
                    self.available_tools.append({
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": {
                            "properties": resolved_properties,
                            "required": tool.inputSchema.get("required", [])
                        },
                    })
    
    def _get_tools_description(self) -> str:
        """LLM에게 제공할 도구 설명 생성"""
        if not self.available_tools:
            return ""
        
        tools_desc = "\n## Available Tools\n\n"
        for tool in self.available_tools:
            tools_desc += f"**{tool['name']}** (Server: {tool['server']})\n"
            tools_desc += f"- {tool['description']}\n"
            props = tool['input_schema'].get('properties', {})
            if props:
                # 이제 완전히 해소된 스키마를 보여줌
                tools_desc += f"- Parameters: {json.dumps(props, ensure_ascii=False, indent=2)}\n"
            tools_desc += "\n"
        
        return tools_desc
```

### LLM이 생성하는 올바른 도구 호출
```json
{
  "tool_call": {
    "server": "nara-api",
    "tool": "get_contstruction_bids",
    "arguments": {
      "page": "1",
      "rows": "10",
      "inqryBgnDt": "202512270000",
      "inqryEndDt": "202512282315",
      "bidNtceNo": ""
    }
  }
}
```

---

## 5. 대안 및 베스트 프랙티스

### 대안 1: BaseModel 사용 안 함
```python
# Pydantic 모델 대신 직접 파라미터 정의
@app.tool()
def get_contstruction_bids(
    page: str = "1",
    rows: str = "1",
    inqryBgnDt: str = "",
    inqryEndDt: str = "",
    bidNtceNo: str = ""
):
    """입찰 정보 조회"""
    pass
```

**장점:**
- ✅ `$ref` 없이 바로 properties에 나옴
- ✅ 간단한 경우 빠른 구현

**단점:**
- ❌ 타입 검증 불가능
- ❌ 기본값 처리 복잡
- ❌ 문서화 자동 생성 제한
- ❌ 여러 곳에서 재사용 시 중복 코드

### 대안 2: Dict 타입 사용
```python
@app.tool()
def get_contstruction_bids(params: dict):
    """입찰 정보 조회"""
    # 수동으로 검증 필요
    page = params.get("page", "1")
    # ...
```

**문제점:**
- ❌ 타입 안전성 완전히 상실
- ❌ IDE 자동완성 불가능
- ❌ 런타임 에러 가능성 증가

### 권장 방식: BaseModel + `$ref` 해소 ✅
```python
# MCP 서버 정의
class BidSearchParams(BaseModel):
    page: str = Field(default="1")
    rows: str = Field(default="1")
    # ... with validation

@app.tool()
def get_contstruction_bids(params: BidSearchParams):
    pass

# 클라이언트 구현
class MCPClient:
    def _resolve_schema_refs(self, schema, definitions):
        # $ref 해소 로직
        pass
    
    async def _connect_mcp_server(self, ...):
        # 도구 수집 시 스키마 해소
        resolved_properties = self._resolve_schema_refs(...)
```

**장점:**
- ✅ 타입 안전성 (Pydantic 검증)
- ✅ 기본값 처리 자동
- ✅ 문서화 자동 생성
- ✅ 스키마 재사용 가능
- ✅ LLM 통합 완벽 지원

---

## 핵심 정리

### ⚠️ 반드시 기억할 것

1. **Pydantic BaseModel을 MCP 도구에 사용하면 `$ref` 구조가 생성됨**
2. **LLM은 `$ref` 참조를 이해하지 못함**
3. **클라이언트에서 `$ref` 해소가 필수**

### ✅ 체크리스트
```python
# MCP 서버 개발 시
□ BaseModel 사용 시 $ref 생성 인지
□ FastMCP/Pydantic의 schema 구조 이해

# MCP 클라이언트 개발 시
□ _resolve_schema_refs() 함수 구현
□ 도구 수집 시 스키마 해소 적용
□ LLM에게 완전히 해소된 스키마 전달
□ 로그로 해소 결과 확인

# 테스트
□ LLM이 올바른 파라미터 생성하는지 확인
□ 중첩된 $ref도 정상 해소되는지 테스트
```

### 🎯 베스트 프랙티스
```python
# 1. 재사용 가능한 유틸리티로 구현
class SchemaResolver:
    @staticmethod
    def resolve_refs(schema: Any, definitions: Dict) -> Any:
        # 재사용 가능한 해소 로직
        pass

# 2. 로깅 추가
logger.info(f"✅ Schema resolved: {json.dumps(resolved, indent=2)}")

# 3. 에러 핸들링
try:
    resolved = self._resolve_schema_refs(schema, defs)
except Exception as e:
    logger.error(f"❌ Schema resolution failed: {e}")
    # Fallback 처리
```

---

## 참고 자료

- [JSON Schema Specification - $ref](https://json-schema.org/understanding-json-schema/structuring.html#ref)
- [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol Spec](https://spec.modelcontextprotocol.io/)

---

**작성일**: 2025-12-28  
**카테고리**: MCP, LLM Integration  
**태그**: #MCP #Pydantic #BaseModel #JSONSchema #Ref #FastMCP