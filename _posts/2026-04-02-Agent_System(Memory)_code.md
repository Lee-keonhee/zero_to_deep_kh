---
layout: post
title: Agent 시스템 (Memory) 예시 코드
summary: agent 시스템의 주요 요소
author: keonhee
date: 2025-04-02 12:00:00 +0900
category: Agent system
keywords: Agent
permalink: /blog/agent_system_memory_code/
usemathjax: false
thumbnail: /assets/img/posts/agent_system_tree.png
imageNameKey: Agent System
---
![](assets/img/posts/agent_system_memory.png)
# Memory

```python
import time
import math
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────
# 1. 데이터 구조
# ──────────────────────────────────────────

ITEM_TYPE_WEIGHT = {
    "system":  1.0,   # 시스템 프롬프트 — 최고 우선순위 (절대 삭제 안 함)
    "user":    0.8,   # 유저 입력
    "context": 0.6,   # 대화 맥락
    "history": 0.3,   # 오래된 히스토리 — 삭제 1순위
}


@dataclass
class MemoryItem:
    id: str
    content: str
    item_type: str                    # "system" | "user" | "context" | "history"
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    score: float = 0.0

    def token_count(self) -> int:
        """간단히 단어 수로 토큰을 근사 계산"""
        return len(self.content.split())


# ──────────────────────────────────────────
# 2. Priority Scoring
# ──────────────────────────────────────────

def calculate_score(item: MemoryItem, current_time: Optional[float] = None) -> float:
    """
    세 가지 요소로 우선순위 점수를 계산.
      - recency   : 최근에 저장됐을수록 높음
      - frequency : 자주 참조됐을수록 높음
      - type      : 타입별 고정 가중치
    """
    if current_time is None:
        current_time = time.time()

    elapsed = current_time - item.timestamp
    recency   = 1.0 / (1.0 + elapsed)              # 0 ~ 1 사이로 정규화
    frequency = math.log(item.access_count + 1)    # 0에서 시작, 서서히 증가

    type_weight = ITEM_TYPE_WEIGHT.get(item.item_type, 0.5)

    item.score = (recency * 0.4) + (frequency * 0.3) + (type_weight * 0.3)
    return item.score


# ──────────────────────────────────────────
# 3. Truncation Strategy
# ──────────────────────────────────────────

class MemoryManager:
    def __init__(self, token_limit: int):
        self.token_limit   = token_limit
        self.memory_pool:  list[MemoryItem] = []
        self.current_tokens: int = 0

    # ── 아이템 추가 ──────────────────────────
    def add(self, item: MemoryItem) -> None:
        needed = item.token_count()

        if self.current_tokens + needed > self.token_limit:
            print(f"\n⚠️  토큰 한계 초과 — 트런케이션 시작 (필요: {needed}토큰)")
            self._truncate(needed)

        self.memory_pool.append(item)
        self.current_tokens += needed
        print(f"✅ 추가됨 [{item.id}] | 타입: {item.item_type} | 토큰: {needed} | 합계: {self.current_tokens}/{self.token_limit}")

    # ── 아이템 참조 (access_count 증가) ──────
    def access(self, item_id: str) -> Optional[MemoryItem]:
        for item in self.memory_pool:
            if item.id == item_id:
                item.access_count += 1
                calculate_score(item)   # 점수 즉시 갱신
                return item
        return None

    # ── 핵심: 트런케이션 실행 ─────────────────
    def _truncate(self, required_tokens: int) -> None:
        # 1단계 — 전체 점수 재계산
        now = time.time()
        for item in self.memory_pool:
            calculate_score(item, current_time=now)

        # 2단계 — 점수 오름차순 정렬 (낮은 것 = 삭제 우선)
        candidates = sorted(self.memory_pool, key=lambda x: x.score)

        # 3단계 — 필요한 만큼 삭제
        freed = 0
        for item in candidates:
            if freed >= required_tokens:
                break

            if item.item_type == "system":          # system은 절대 보호
                print(f"  🔒 보호됨 [{item.id}] (system 타입)")
                continue

            self.memory_pool.remove(item)
            freed += item.token_count()
            self.current_tokens -= item.token_count()
            print(f"  🗑️  삭제됨 [{item.id}] | 점수: {item.score:.4f} | 확보: {item.token_count()}토큰")

        if freed < required_tokens:
            print("  ⚠️  system 항목 제외 후 충분한 공간을 확보하지 못했습니다.")

    # ── 현재 상태 출력 ────────────────────────
    def status(self) -> None:
        print(f"\n{'─'*55}")
        print(f"📦 메모리 풀 상태 ({self.current_tokens}/{self.token_limit} 토큰)")
        print(f"{'─'*55}")
        now = time.time()
        sorted_pool = sorted(self.memory_pool, key=lambda x: x.score, reverse=True)
        for item in sorted_pool:
            calculate_score(item, current_time=now)
            print(f"  [{item.id}] {item.item_type:<8} | "
                  f"점수: {item.score:.4f} | "
                  f"토큰: {item.token_count():>3} | "
                  f"참조: {item.access_count}회 | "
                  f"{item.content[:30]}...")
        print(f"{'─'*55}\n")


# ──────────────────────────────────────────
# 4. 실행 예제
# ──────────────────────────────────────────

if __name__ == "__main__":
    # 토큰 한계를 50으로 설정 (데모용으로 작게)
    manager = MemoryManager(token_limit=50)

    # 시스템 프롬프트 추가 (보호 대상)
    manager.add(MemoryItem("sys-1", "You are a helpful assistant. Always be concise and accurate.", "system"))

    # 유저 대화 추가
    manager.add(MemoryItem("usr-1", "Hello can you help me write a Python function", "user"))
    manager.add(MemoryItem("ctx-1", "The user is working on a data pipeline project using pandas", "context"))
    manager.add(MemoryItem("his-1", "Earlier the user asked about sorting algorithms bubble sort", "history"))
    manager.add(MemoryItem("his-2", "The user mentioned they prefer functional programming style code", "history"))

    # usr-1을 여러 번 참조 → 점수 올라감
    manager.access("usr-1")
    manager.access("usr-1")
    manager.access("usr-1")

    manager.status()

    # 새 항목 추가 시 토큰 한계 초과 → 트런케이션 발동
    print("=== 새 항목 추가 (트런케이션 발동 예상) ===")
    manager.add(MemoryItem("usr-2", "Now I need to write a function that filters rows by date range efficiently", "user"))

    manager.status()
```