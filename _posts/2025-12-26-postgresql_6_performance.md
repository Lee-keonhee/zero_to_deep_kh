---
layout: post
title: PostgreSQL 완전 정복 6단계 - 성능 최적화 완벽 가이드
summary: 인덱스 전략, EXPLAIN 분석, 쿼리 튜닝으로 PostgreSQL 성능 극대화하기
author: keonhee
date: 2025-12-26 09:00:00 +0900
category: Database
keywords: PostgreSQL, Performance, Index, EXPLAIN, Query Optimization, Tuning
permalink: /blog/postgresql_performance/
usemathjax: false
thumbnail: /assets/img/posts/postgresql_performance.png
---

# PostgreSQL 3단계: 데이터 조작 (CRUD)

# PostgreSQL 6단계: 성능 최적화

## 목차
1. [인덱스 생성 및 관리](#1-인덱스-생성-및-관리)
2. [쿼리 실행 계획 (EXPLAIN)](#2-쿼리-실행-계획-explain)
3. [성능 튜닝 기법](#3-성능-튜닝-기법)

---

## 1. 인덱스 생성 및 관리

### 인덱스란?

인덱스는 책의 색인과 같이 데이터를 빠르게 찾을 수 있도록 도와주는 데이터베이스 객체입니다.

**장점:**
- SELECT 쿼리 속도 향상
- WHERE, JOIN, ORDER BY 성능 개선
- 고유성 보장 (UNIQUE 인덱스)

**단점:**
- INSERT, UPDATE, DELETE 속도 저하
- 추가 저장 공간 필요
- 유지보수 오버헤드

### B-tree 인덱스 (기본)

```sql
-- 단일 열 인덱스
CREATE INDEX idx_products_price ON products(price);

-- 여러 열 복합 인덱스
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);

-- 내림차순 인덱스
CREATE INDEX idx_products_price_desc ON products(price DESC);

-- 부분 인덱스 (조건부)
CREATE INDEX idx_active_users ON users(email)
WHERE active = true;

-- 표현식 인덱스
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- 이름 지정 안 하면 자동 생성
CREATE INDEX ON products(category);
```

### UNIQUE 인덱스

```sql
-- 고유 인덱스 생성
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- 복합 고유 인덱스
CREATE UNIQUE INDEX idx_product_reviews 
ON product_reviews(user_id, product_id);

-- 부분 고유 인덱스
CREATE UNIQUE INDEX idx_active_usernames ON users(username)
WHERE active = true;
```

### 다양한 인덱스 타입

**Hash 인덱스**
```sql
-- 등호(=) 비교에만 사용
CREATE INDEX idx_users_id_hash ON users USING HASH(user_id);
```

**GiST 인덱스 (공간 데이터, 전문 검색)**
```sql
-- PostGIS 지리 데이터용
CREATE INDEX idx_locations_geom ON locations USING GIST(geom);

-- 범위 검색
CREATE INDEX idx_events_period ON events USING GIST(event_period);
```

**GIN 인덱스 (JSON, 배열, 전문 검색)**
```sql
-- JSON 데이터
CREATE INDEX idx_products_attributes ON products USING GIN(attributes);

-- 배열
CREATE INDEX idx_posts_tags ON posts USING GIN(tags);

-- 전문 검색
CREATE INDEX idx_articles_content ON articles USING GIN(to_tsvector('english', content));
```

**BRIN 인덱스 (대용량 정렬 데이터)**
```sql
-- 시계열 데이터에 효과적
CREATE INDEX idx_logs_timestamp ON logs USING BRIN(created_at);
```

### 인덱스 관리

```sql
-- 인덱스 목록 확인
\di

-- 특정 테이블의 인덱스
\d products

-- SQL로 확인
SELECT 
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'products';

-- 인덱스 크기 확인
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'products';

-- 인덱스 사용 통계
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE tablename = 'products';

-- 사용되지 않는 인덱스 찾기
SELECT 
    schemaname || '.' || tablename AS table,
    indexname AS index,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size,
    idx_scan AS scans
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexname::regclass) DESC;
```

### 인덱스 삭제 및 재생성

```sql
-- 인덱스 삭제
DROP INDEX idx_products_price;

-- 존재할 경우에만 삭제
DROP INDEX IF EXISTS idx_products_price;

-- 동시 삭제 (테이블 잠금 없이)
DROP INDEX CONCURRENTLY idx_products_price;

-- 인덱스 재생성
REINDEX INDEX idx_products_price;

-- 테이블의 모든 인덱스 재생성
REINDEX TABLE products;

-- 데이터베이스의 모든 인덱스 재생성
REINDEX DATABASE mydb;

-- 동시 재생성 (권장)
CREATE INDEX CONCURRENTLY idx_products_price_new ON products(price);
DROP INDEX CONCURRENTLY idx_products_price;
ALTER INDEX idx_products_price_new RENAME TO idx_products_price;
```

### 인덱스 전략

```sql
-- 1. 자주 조회되는 열
CREATE INDEX idx_orders_status ON orders(status);

-- 2. WHERE 절에 자주 사용되는 열
CREATE INDEX idx_products_category ON products(category);

-- 3. JOIN에 사용되는 외래 키
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- 4. ORDER BY에 사용되는 열
CREATE INDEX idx_orders_date ON orders(order_date DESC);

-- 5. 복합 인덱스 순서 (선택도가 높은 열을 먼저)
CREATE INDEX idx_users_city_age ON users(city, age);  -- city가 선택도 높음

-- 6. 커버링 인덱스 (모든 필요한 열 포함)
CREATE INDEX idx_orders_cover ON orders(customer_id, order_date, total_amount);
```

---

## 2. 쿼리 실행 계획 (EXPLAIN)

### EXPLAIN 기본

```sql
-- 기본 실행 계획
EXPLAIN SELECT * FROM products WHERE price > 100;

-- 실제 실행 포함
EXPLAIN ANALYZE SELECT * FROM products WHERE price > 100;

-- 상세 정보
EXPLAIN (ANALYZE, VERBOSE) SELECT * FROM products WHERE price > 100;

-- 버퍼 정보 포함
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM products WHERE price > 100;

-- 모든 옵션
EXPLAIN (
    ANALYZE true,
    VERBOSE true,
    COSTS true,
    BUFFERS true,
    TIMING true,
    SUMMARY true,
    FORMAT JSON
) SELECT * FROM products WHERE price > 100;
```

### 실행 계획 읽기

```sql
EXPLAIN ANALYZE
SELECT 
    c.customer_name,
    o.order_id,
    o.total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date > '2024-01-01';

-- 출력 해석:
-- Seq Scan: 순차 스캔 (전체 테이블 읽기) - 느림
-- Index Scan: 인덱스 스캔 - 빠름
-- Index Only Scan: 인덱스만 사용 - 매우 빠름
-- Bitmap Index Scan: 비트맵 인덱스 스캔
-- Hash Join: 해시 조인
-- Nested Loop: 중첩 루프 조인
-- Merge Join: 병합 조인

-- cost=시작비용..총비용
-- rows=예상 행 수
-- actual time=실제 시간
```

### 성능 문제 진단

```sql
-- 순차 스캔 문제
EXPLAIN ANALYZE
SELECT * FROM products WHERE category = 'Electronics';
-- Seq Scan → 인덱스 필요

-- 해결
CREATE INDEX idx_products_category ON products(category);

-- 조인 성능 문제
EXPLAIN ANALYZE
SELECT *
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id;
-- Nested Loop 또는 Hash Join

-- 외래 키에 인덱스 생성
CREATE INDEX idx_order_items_order ON order_items(order_id);

-- 정렬 성능
EXPLAIN ANALYZE
SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;
-- Sort 또는 Index Scan

-- 인덱스로 해결
CREATE INDEX idx_orders_date ON orders(order_date DESC);
```

### 실제 예제 분석

```sql
-- 복잡한 쿼리 분석
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    c.customer_name,
    COUNT(o.order_id) AS order_count,
    SUM(o.total_amount) AS total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.city = 'Seoul'
  AND o.order_date >= '2024-01-01'
GROUP BY c.customer_id, c.customer_name
HAVING SUM(o.total_amount) > 1000
ORDER BY total_spent DESC
LIMIT 10;

-- 최적화 포인트 체크:
-- 1. customers.city 인덱스
-- 2. orders.order_date 인덱스  
-- 3. orders.customer_id 인덱스
-- 4. 커버링 인덱스 고려
```

---

## 3. 성능 튜닝 기법

### 쿼리 최적화

```sql
-- 1. 필요한 열만 선택
-- 나쁜 예
SELECT * FROM products;

-- 좋은 예
SELECT product_id, product_name, price FROM products;

-- 2. LIMIT 사용
SELECT * FROM orders ORDER BY order_date DESC LIMIT 100;

-- 3. EXISTS 사용 (IN 대신)
-- 느림
SELECT * FROM customers 
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE total_amount > 1000
);

-- 빠름
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id 
      AND o.total_amount > 1000
);

-- 4. UNION ALL 사용 (UNION 대신)
SELECT product_name FROM products WHERE category = 'Electronics'
UNION ALL  -- 중복 제거 안 함 (빠름)
SELECT product_name FROM archived_products WHERE category = 'Electronics';

-- 5. JOIN 대신 서브쿼리 (경우에 따라)
-- 작은 결과 집합일 때 서브쿼리가 더 빠를 수 있음
```

### 테이블 설계 최적화

```sql
-- 1. 적절한 데이터 타입 선택
-- 나쁜 예
CREATE TABLE users (
    age VARCHAR(10)  -- 문자열로 숫자 저장
);

-- 좋은 예
CREATE TABLE users (
    age INTEGER
);

-- 2. NOT NULL 사용 (가능한 경우)
CREATE TABLE products (
    product_name VARCHAR(200) NOT NULL,  -- NULL 체크 불필요
    price DECIMAL(10, 2) NOT NULL
);

-- 3. 정규화 vs 반정규화
-- 정규화: 중복 제거, 무결성 향상
-- 반정규화: 조회 성능 향상 (계산된 열 추가)

CREATE TABLE order_summary (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    total_amount DECIMAL(10, 2),
    item_count INTEGER  -- 반정규화 (계산하지 않고 저장)
);
```

### 파티셔닝

```sql
-- 범위 파티셔닝 (날짜별)
CREATE TABLE orders_partitioned (
    order_id SERIAL,
    customer_id INTEGER,
    order_date DATE NOT NULL,
    total_amount DECIMAL(10, 2)
) PARTITION BY RANGE (order_date);

-- 파티션 생성
CREATE TABLE orders_2024_q1 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- 리스트 파티셔닝 (카테고리별)
CREATE TABLE products_partitioned (
    product_id SERIAL,
    product_name VARCHAR(200),
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10, 2)
) PARTITION BY LIST (category);

CREATE TABLE products_electronics PARTITION OF products_partitioned
    FOR VALUES IN ('Electronics', 'Computers');

CREATE TABLE products_books PARTITION OF products_partitioned
    FOR VALUES IN ('Books', 'Ebooks');
```

### VACUUM과 ANALYZE

```sql
-- VACUUM: 삭제된 행의 공간 회수
VACUUM products;

-- VACUUM FULL: 테이블 재구성 (느림, 잠금)
VACUUM FULL products;

-- ANALYZE: 통계 정보 업데이트
ANALYZE products;

-- 둘 다 실행
VACUUM ANALYZE products;

-- 자동 VACUUM 설정 확인
SHOW autovacuum;

-- 마지막 VACUUM 시간 확인
SELECT 
    schemaname,
    relname,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE relname = 'products';
```

### 연결 풀링

```sql
-- pg_bouncer 설정 예제
-- /etc/pgbouncer/pgbouncer.ini

[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25

-- 애플리케이션은 6432 포트로 연결
-- psql -h localhost -p 6432 -U user mydb
```

### 캐싱 전략

```sql
-- 1. 구체화된 뷰 (Materialized View)
CREATE MATERIALIZED VIEW sales_summary AS
SELECT 
    DATE_TRUNC('day', order_date) AS day,
    COUNT(*) AS order_count,
    SUM(total_amount) AS daily_revenue
FROM orders 
GROUP BY DATE_TRUNC('day', order_date);

-- 인덱스 추가
CREATE INDEX ON sales_summary(day);

-- 정기적으로 갱신
REFRESH MATERIALIZED VIEW sales_summary;

-- 2. 애플리케이션 레벨 캐싱
-- Redis, Memcached 사용
```

### 배치 처리

```sql
-- 1. 대량 INSERT
-- 나쁜 예
BEGIN;
INSERT INTO logs (message) VALUES ('log1');
INSERT INTO logs (message) VALUES ('log2');
-- 1000번 반복
COMMIT;

-- 좋은 예
INSERT INTO logs (message) VALUES
    ('log1'),
    ('log2'),
    -- ... 1000개
    ('log1000');

-- 2. COPY 명령 (가장 빠름)
COPY logs (message) FROM '/path/to/data.csv' CSV;

-- 3. 대량 UPDATE
-- 나쁜 예
UPDATE products SET stock = stock - 1 WHERE product_id = 1;
UPDATE products SET stock = stock - 1 WHERE product_id = 2;

-- 좋은 예
UPDATE products
SET stock = stock - v.quantity
FROM (VALUES 
    (1, 1),
    (2, 2),
    (3, 5)
) AS v(product_id, quantity)
WHERE products.product_id = v.product_id;
```

### 모니터링

```sql
-- 1. 느린 쿼리 찾기
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- 2. 테이블 크기 확인
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 3. 활성 연결 확인
SELECT 
    datname,
    usename,
    application_name,
    client_addr,
    state,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- 4. 인덱스 사용률
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC, pg_relation_size(indexname::regclass) DESC;
```

### 성능 최적화 체크리스트

```sql
-- ✅ 인덱스
-- 1. WHERE 절에 자주 사용되는 열
-- 2. JOIN에 사용되는 외래 키
-- 3. ORDER BY에 사용되는 열
-- 4. 복합 인덱스 (자주 함께 사용되는 열)

-- ✅ 쿼리
-- 1. 필요한 열만 SELECT
-- 2. LIMIT 사용
-- 3. EXISTS 대신 IN (소량일 때)
-- 4. 적절한 JOIN 타입

-- ✅ 테이블 설계
-- 1. 적절한 데이터 타입
-- 2. NOT NULL 제약
-- 3. 정규화
-- 4. 파티셔닝 (대용량)

-- ✅ 유지보수
-- 1. 정기적인 VACUUM ANALYZE
-- 2. 통계 정보 업데이트
-- 3. 사용하지 않는 인덱스 제거
-- 4. 쿼리 모니터링
```

---

## 실전 예제

### 전자상거래 성능 최적화

```sql
-- 1. 인덱스 전략
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date DESC);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_products_category ON products(category);

-- 2. 복합 인덱스
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date DESC);

-- 3. 부분 인덱스
CREATE INDEX idx_active_orders ON orders(order_id)
WHERE status IN ('pending', 'processing');

-- 4. 커버링 인덱스
CREATE INDEX idx_orders_summary ON orders(customer_id, order_date, total_amount);

-- 5. 구체화된 뷰
CREATE MATERIALIZED VIEW daily_sales AS
SELECT 
    DATE(order_date) AS sale_date,
    COUNT(*) AS order_count,
    SUM(total_amount) AS revenue
FROM orders 
WHERE status = 'completed'
GROUP BY DATE(order_date);

CREATE INDEX ON daily_sales(sale_date);

-- 6. 최적화된 쿼리
SELECT 
    p.product_name,
    SUM(oi.quantity) AS total_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
  AND o.status = 'completed'
GROUP BY p.product_id, p.product_name
ORDER BY total_sold DESC
LIMIT 10;
```

---

## 다음 단계

6단계를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ 효과적인 인덱스 생성 및 관리
✅ EXPLAIN으로 쿼리 성능 분석
✅ 다양한 성능 튜닝 기법 적용

**다음 학습**: 7단계 - 고급 기능
- 뷰 (VIEW)와 구체화된 뷰
- 트랜잭션과 격리 수준
- 함수와 프로시저
- 트리거
- CTE (Common Table Expressions)
- 윈도우 함수
