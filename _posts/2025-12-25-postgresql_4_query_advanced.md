---
layout: post
title: PostgreSQL 완전 정복 4단계 - 쿼리 심화 기법
summary: WHERE 조건, 정렬, 집계함수, GROUP BY, 서브쿼리까지 고급 쿼리 작성법
author: keonhee
date: 2025-12-25 09:00:00 +0900
category: Database
keywords: PostgreSQL, Query, WHERE, GROUP BY, Aggregate Functions, Subquery
permalink: /blog/postgresql_query_advanced/
usemathjax: false
thumbnail: /assets/img/posts/postgresql.png
imageNameKey: postgresql
---


# PostgreSQL 4단계: 데이터 조회 심화

## 목차
1. [WHERE 절과 조건 연산자](#1-where-절과-조건-연산자)
2. [정렬 (ORDER BY)과 제한 (LIMIT, OFFSET)](#2-정렬-order-by과-제한-limit-offset)
3. [집계 함수](#3-집계-함수)
4. [GROUP BY와 HAVING](#4-group-by와-having)
5. [서브쿼리](#5-서브쿼리)

---

## 1. WHERE 절과 조건 연산자

### 비교 연산자

```sql
-- 같음
SELECT * FROM products WHERE price = 100;

-- 같지 않음
SELECT * FROM products WHERE price != 100;
SELECT * FROM products WHERE price <> 100;  -- != 와 동일

-- 크다/작다
SELECT * FROM products WHERE price > 100;
SELECT * FROM products WHERE price < 100;
SELECT * FROM products WHERE price >= 100;
SELECT * FROM products WHERE price <= 100;
```

### 논리 연산자

```sql
-- AND: 모든 조건이 참
SELECT * FROM products 
WHERE price > 100 AND stock > 10;

SELECT * FROM products 
WHERE category = 'Electronics' 
  AND price BETWEEN 500 AND 1000
  AND stock > 0;

-- OR: 하나라도 참
SELECT * FROM products 
WHERE category = 'Electronics' OR category = 'Books';

SELECT * FROM products 
WHERE price < 50 OR stock < 5;

-- NOT: 조건 부정
SELECT * FROM products 
WHERE NOT category = 'Electronics';

SELECT * FROM products 
WHERE NOT (price > 100 AND stock < 10);

-- 복합 조건 (괄호 사용)
SELECT * FROM products 
WHERE (category = 'Electronics' AND price > 500)
   OR (category = 'Books' AND price < 50);
```

### BETWEEN 연산자

```sql
-- 범위 검색 (양쪽 끝 포함)
SELECT * FROM products 
WHERE price BETWEEN 100 AND 500;
-- price >= 100 AND price <= 500 와 동일

-- 날짜 범위
SELECT * FROM orders 
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';

-- NOT BETWEEN
SELECT * FROM products 
WHERE price NOT BETWEEN 100 AND 500;
```

### IN 연산자

```sql
-- 목록에서 매칭
SELECT * FROM products 
WHERE category IN ('Electronics', 'Books', 'Toys');
-- category = 'Electronics' OR category = 'Books' OR category = 'Toys' 와 동일

-- 숫자 목록
SELECT * FROM students 
WHERE age IN (18, 19, 20, 21);

-- NOT IN
SELECT * FROM products 
WHERE category NOT IN ('Discontinued', 'Out of Stock');

-- 서브쿼리와 함께 사용
SELECT * FROM products 
WHERE product_id IN (
    SELECT product_id FROM order_items 
    WHERE quantity > 10
);
```

### LIKE 패턴 매칭

```sql
-- % : 0개 이상의 문자
-- _ : 정확히 1개의 문자

-- 시작 패턴
SELECT * FROM products 
WHERE product_name LIKE 'Laptop%';  -- Laptop으로 시작

-- 종료 패턴
SELECT * FROM products 
WHERE product_name LIKE '%Pro';  -- Pro로 끝남

-- 포함 패턴
SELECT * FROM products 
WHERE product_name LIKE '%Dell%';  -- Dell 포함

-- 위치 지정
SELECT * FROM products 
WHERE product_code LIKE 'A_C%';  -- A로 시작, 3번째가 C

-- 대소문자 구분 없음 (ILIKE)
SELECT * FROM products 
WHERE product_name ILIKE '%laptop%';  -- PostgreSQL 전용

-- NOT LIKE
SELECT * FROM products 
WHERE product_name NOT LIKE '%Refurbished%';

-- 이스케이프 문자
SELECT * FROM products 
WHERE description LIKE '%50\% off%' ESCAPE '\';  -- % 문자 자체를 찾기
```

### NULL 처리

```sql
-- NULL 체크
SELECT * FROM products 
WHERE description IS NULL;

-- NOT NULL 체크
SELECT * FROM products 
WHERE description IS NOT NULL;

-- NULL 처리 함수
SELECT 
    product_name,
    COALESCE(description, 'No description available') AS description
FROM products;

-- NULLIF (두 값이 같으면 NULL 반환)
SELECT NULLIF(stock, 0) FROM products;  -- stock이 0이면 NULL

-- NULL과 비교 연산 주의
SELECT * FROM products WHERE price = NULL;  -- 잘못된 방법, 항상 false
SELECT * FROM products WHERE price IS NULL;  -- 올바른 방법
```

### 정규 표현식

```sql
-- ~ : 정규식 매칭 (대소문자 구분)
SELECT * FROM products 
WHERE product_name ~ '^[A-Z]';  -- 대문자로 시작

-- ~* : 정규식 매칭 (대소문자 구분 없음)
SELECT * FROM products 
WHERE product_name ~* 'laptop|desktop';  -- laptop 또는 desktop 포함

-- !~ : 정규식 불일치
SELECT * FROM products 
WHERE product_code !~ '[0-9]';  -- 숫자를 포함하지 않음

-- SIMILAR TO (SQL 표준)
SELECT * FROM products 
WHERE product_name SIMILAR TO '%(Laptop|Desktop)%';
```

---

## 2. 정렬 (ORDER BY)과 제한 (LIMIT, OFFSET)

### ORDER BY 기본

```sql
-- 오름차순 (기본값)
SELECT * FROM products ORDER BY price;
SELECT * FROM products ORDER BY price ASC;

-- 내림차순
SELECT * FROM products ORDER BY price DESC;

-- 문자열 정렬
SELECT * FROM products ORDER BY product_name;  -- 알파벳 순
SELECT * FROM products ORDER BY product_name DESC;  -- 역순

-- 날짜 정렬
SELECT * FROM orders ORDER BY order_date DESC;  -- 최신순
SELECT * FROM orders ORDER BY created_at ASC;  -- 오래된 순
```

### 여러 열로 정렬

```sql
-- 우선순위: category 오름차순, 그 다음 price 내림차순
SELECT * FROM products 
ORDER BY category ASC, price DESC;

-- 3개 이상의 열
SELECT * FROM products 
ORDER BY category, stock DESC, price ASC;

-- 실용 예제: 카테고리별 가격 높은 순
SELECT category, product_name, price
FROM products 
ORDER BY category, price DESC;
```

### NULL 값 정렬

```sql
-- NULL을 먼저 표시
SELECT * FROM products 
ORDER BY description NULLS FIRST;

-- NULL을 나중에 표시
SELECT * FROM products 
ORDER BY description NULLS LAST;

-- 기본 동작
-- ASC: NULLS LAST (NULL이 마지막)
-- DESC: NULLS FIRST (NULL이 처음)

SELECT * FROM products 
ORDER BY description DESC NULLS LAST;
```

### 표현식으로 정렬

```sql
-- 계산 결과로 정렬
SELECT 
    product_name,
    price,
    stock,
    price * stock AS total_value
FROM products 
ORDER BY price * stock DESC;

-- 문자열 길이로 정렬
SELECT product_name 
FROM products 
ORDER BY LENGTH(product_name);

-- CASE 문으로 정렬
SELECT * FROM products 
ORDER BY 
    CASE category
        WHEN 'Electronics' THEN 1
        WHEN 'Books' THEN 2
        WHEN 'Toys' THEN 3
        ELSE 4
    END;

-- 조건부 정렬
SELECT * FROM products 
ORDER BY 
    CASE 
        WHEN stock = 0 THEN 0  -- 품절 제품을 마지막에
        ELSE 1
    END DESC,
    price ASC;
```

### LIMIT - 결과 개수 제한

```sql
-- 상위 10개
SELECT * FROM products 
ORDER BY price DESC 
LIMIT 10;

-- 최저가 5개 상품
SELECT * FROM products 
ORDER BY price ASC 
LIMIT 5;

-- 최근 주문 20개
SELECT * FROM orders 
ORDER BY order_date DESC 
LIMIT 20;
```

### OFFSET - 건너뛰기

```sql
-- 11번째부터 10개 (11~20)
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10 OFFSET 10;

-- 처음 10개 건너뛰기
SELECT * FROM products 
ORDER BY created_at DESC 
OFFSET 10;

-- OFFSET만 사용 (모든 결과를 10개 건너뛴 후 조회)
SELECT * FROM products 
ORDER BY price 
OFFSET 5;
```

### 페이징 구현

```sql
-- 페이지 크기: 10, 페이지 번호: 1 (1~10)
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10 OFFSET 0;

-- 페이지 2 (11~20)
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10 OFFSET 10;

-- 페이지 3 (21~30)
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10 OFFSET 20;

-- 일반 공식: OFFSET = (페이지번호 - 1) * 페이지크기
-- 페이지 N 가져오기
SELECT * FROM products 
ORDER BY product_id 
LIMIT 10 OFFSET (N - 1) * 10;

-- 페이징 개선 (OFFSET 대신 키 기반)
-- 마지막으로 본 ID 이후의 데이터 가져오기
SELECT * FROM products 
WHERE product_id > :last_seen_id
ORDER BY product_id 
LIMIT 10;
```

### FETCH (SQL 표준)

```sql
-- LIMIT 대신 FETCH 사용 (SQL 표준)
SELECT * FROM products 
ORDER BY price 
FETCH FIRST 10 ROWS ONLY;

-- OFFSET과 함께
SELECT * FROM products 
ORDER BY price 
OFFSET 5 ROWS 
FETCH NEXT 10 ROWS ONLY;
```

---

## 3. 집계 함수

### COUNT - 개수 세기

```sql
-- 전체 행 개수
SELECT COUNT(*) FROM products;

-- NULL이 아닌 값 개수
SELECT COUNT(description) FROM products;

-- 중복 제거 개수
SELECT COUNT(DISTINCT category) FROM products;

-- 조건부 개수
SELECT COUNT(*) FROM products WHERE price > 100;

-- 여러 COUNT
SELECT 
    COUNT(*) AS total,
    COUNT(description) AS with_description,
    COUNT(DISTINCT category) AS categories
FROM products;
```

### SUM - 합계

```sql
-- 전체 재고 합계
SELECT SUM(stock) FROM products;

-- 카테고리별 재고 합계
SELECT SUM(stock) FROM products WHERE category = 'Electronics';

-- 계산 결과의 합계
SELECT SUM(price * stock) AS total_inventory_value FROM products;

-- NULL 처리
SELECT SUM(COALESCE(stock, 0)) FROM products;
```

### AVG - 평균

```sql
-- 평균 가격
SELECT AVG(price) FROM products;

-- 반올림
SELECT ROUND(AVG(price), 2) FROM products;

-- 소수점 처리
SELECT AVG(price)::NUMERIC(10,2) FROM products;

-- 조건부 평균
SELECT AVG(price) FROM products WHERE stock > 0;

-- NULL 제외 (기본 동작)
SELECT AVG(price) FROM products;  -- NULL은 자동으로 제외됨
```

### MAX / MIN - 최대값 / 최소값

```sql
-- 최고가
SELECT MAX(price) FROM products;

-- 최저가
SELECT MIN(price) FROM products;

-- 날짜 최대/최소
SELECT 
    MIN(created_at) AS earliest,
    MAX(created_at) AS latest
FROM orders;

-- 문자열 최대/최소 (알파벳 순)
SELECT 
    MIN(product_name) AS first_alphabetically,
    MAX(product_name) AS last_alphabetically
FROM products;

-- MAX/MIN 행 전체 가져오기
SELECT * FROM products 
WHERE price = (SELECT MAX(price) FROM products);
```

### 여러 집계 함수 조합

```sql
SELECT 
    COUNT(*) AS total_products,
    COUNT(DISTINCT category) AS categories,
    AVG(price)::NUMERIC(10,2) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    SUM(stock) AS total_stock,
    SUM(price * stock)::NUMERIC(12,2) AS total_value
FROM products;

-- 조건부 집계
SELECT 
    COUNT(*) AS total,
    COUNT(CASE WHEN stock > 0 THEN 1 END) AS in_stock,
    COUNT(CASE WHEN stock = 0 THEN 1 END) AS out_of_stock,
    AVG(CASE WHEN stock > 0 THEN price END) AS avg_price_in_stock
FROM products;
```

### STRING_AGG - 문자열 집계

```sql
-- 모든 카테고리를 쉼표로 구분하여 나열
SELECT STRING_AGG(DISTINCT category, ', ') AS all_categories
FROM products;

-- 순서 지정
SELECT STRING_AGG(product_name, ', ' ORDER BY price DESC) AS products
FROM products 
WHERE category = 'Electronics';
```

### ARRAY_AGG - 배열 집계

```sql
-- 모든 가격을 배열로
SELECT ARRAY_AGG(price) AS all_prices FROM products;

-- 중복 제거 및 정렬
SELECT ARRAY_AGG(DISTINCT category ORDER BY category) AS categories
FROM products;
```

---

## 4. GROUP BY와 HAVING

### GROUP BY 기본

```sql
-- 카테고리별 개수
SELECT 
    category,
    COUNT(*) AS count
FROM products 
GROUP BY category;

-- 카테고리별 평균 가격
SELECT 
    category,
    AVG(price)::NUMERIC(10,2) AS avg_price
FROM products 
GROUP BY category;

-- 여러 집계
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price)::NUMERIC(10,2) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    SUM(stock) AS total_stock
FROM products 
GROUP BY category;
```

### 여러 열로 GROUP BY

```sql
-- 카테고리와 재고 상태별 그룹화
SELECT 
    category,
    CASE 
        WHEN stock = 0 THEN 'Out of Stock'
        WHEN stock < 10 THEN 'Low Stock'
        ELSE 'In Stock'
    END AS stock_status,
    COUNT(*) AS count
FROM products 
GROUP BY category, stock_status
ORDER BY category, stock_status;

-- 날짜별 그룹화
SELECT 
    DATE(order_date) AS order_day,
    COUNT(*) AS order_count,
    SUM(total_amount) AS daily_total
FROM orders 
GROUP BY DATE(order_date)
ORDER BY order_day DESC;
```

### HAVING - 그룹 필터링

```sql
-- 상품이 5개 이상인 카테고리만
SELECT 
    category,
    COUNT(*) AS count
FROM products 
GROUP BY category 
HAVING COUNT(*) >= 5;

-- 평균 가격이 100 이상인 카테고리
SELECT 
    category,
    AVG(price)::NUMERIC(10,2) AS avg_price
FROM products 
GROUP BY category 
HAVING AVG(price) > 100;

-- 여러 HAVING 조건
SELECT 
    category,
    COUNT(*) AS count,
    AVG(price)::NUMERIC(10,2) AS avg_price
FROM products 
GROUP BY category 
HAVING COUNT(*) > 3 
   AND AVG(price) > 50
ORDER BY avg_price DESC;
```

### WHERE vs HAVING

```sql
-- WHERE: 그룹화 전 필터링 (개별 행)
-- HAVING: 그룹화 후 필터링 (그룹)

-- 재고가 있는 상품 중, 카테고리별 평균 가격이 100 이상인 것
SELECT 
    category,
    AVG(price)::NUMERIC(10,2) AS avg_price,
    COUNT(*) AS count
FROM products 
WHERE stock > 0  -- 그룹화 전 필터링
GROUP BY category 
HAVING AVG(price) > 100  -- 그룹화 후 필터링
ORDER BY avg_price DESC;

-- 성능: WHERE을 먼저 사용하여 데이터 양 줄이기
SELECT 
    category,
    COUNT(*) AS count
FROM products 
WHERE price > 50  -- 먼저 필터링 (성능 향상)
GROUP BY category 
HAVING COUNT(*) > 10;
```

### 실전 예제

```sql
-- 예제 1: 월별 매출 통계
SELECT 
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS order_count,
    SUM(total_amount) AS monthly_revenue,
    AVG(total_amount)::NUMERIC(10,2) AS avg_order_value
FROM orders 
WHERE order_date >= '2024-01-01'
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- 예제 2: 상위 카테고리 분석
SELECT 
    category,
    COUNT(*) AS products,
    SUM(stock) AS total_stock,
    SUM(price * stock)::NUMERIC(12,2) AS inventory_value,
    AVG(price)::NUMERIC(10,2) AS avg_price
FROM products 
GROUP BY category 
HAVING COUNT(*) >= 3
ORDER BY inventory_value DESC;

-- 예제 3: 고객별 주문 통계
SELECT 
    customer_id,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_spent,
    AVG(total_amount)::NUMERIC(10,2) AS avg_order,
    MAX(order_date) AS last_order_date
FROM orders 
GROUP BY customer_id 
HAVING COUNT(*) > 5  -- 5번 이상 주문한 고객
ORDER BY total_spent DESC 
LIMIT 10;
```

### GROUP BY 고급 기능

```sql
-- ROLLUP: 소계 및 총계 생성
SELECT 
    category,
    COUNT(*) AS count,
    SUM(stock) AS total_stock
FROM products 
GROUP BY ROLLUP(category)
ORDER BY category NULLS LAST;

-- CUBE: 모든 조합의 소계
SELECT 
    category,
    CASE WHEN stock > 0 THEN 'In Stock' ELSE 'Out' END AS status,
    COUNT(*) AS count
FROM products 
GROUP BY CUBE(category, status)
ORDER BY category, status;

-- GROUPING SETS: 특정 그룹만 지정
SELECT 
    category,
    COUNT(*) AS count
FROM products 
GROUP BY GROUPING SETS (
    (category),  -- 카테고리별
    ()           -- 전체
);
```

---

## 5. 서브쿼리

### 스칼라 서브쿼리 (단일 값)

```sql
-- SELECT 절의 서브쿼리
SELECT 
    product_name,
    price,
    (SELECT AVG(price) FROM products) AS avg_price,
    price - (SELECT AVG(price) FROM products) AS price_diff
FROM products;

-- WHERE 절의 서브쿼리
SELECT * FROM products 
WHERE price > (SELECT AVG(price) FROM products);

-- 최고가 상품 찾기
SELECT * FROM products 
WHERE price = (SELECT MAX(price) FROM products);
```

### IN 서브쿼리

```sql
-- 주문된 상품만 조회
SELECT * FROM products 
WHERE product_id IN (
    SELECT DISTINCT product_id 
    FROM order_items
);

-- NOT IN: 한 번도 주문되지 않은 상품
SELECT * FROM products 
WHERE product_id NOT IN (
    SELECT product_id 
    FROM order_items 
    WHERE product_id IS NOT NULL
);

-- 여러 조건
SELECT * FROM products 
WHERE category IN (
    SELECT category 
    FROM products 
    GROUP BY category 
    HAVING AVG(price) > 100
);
```

### EXISTS 서브쿼리

```sql
-- 주문 이력이 있는 고객
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id
);

-- NOT EXISTS: 주문 이력이 없는 고객
SELECT * FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id
);

-- EXISTS vs IN 성능
-- EXISTS가 더 빠른 경우가 많음 (첫 번째 매칭에서 중단)
SELECT * FROM products p
WHERE EXISTS (
    SELECT 1 FROM order_items oi 
    WHERE oi.product_id = p.product_id 
    LIMIT 1
);
```

### FROM 절 서브쿼리 (인라인 뷰)

```sql
-- 카테고리별 평균보다 비싼 상품
SELECT 
    p.*,
    cat_avg.avg_price
FROM products p
JOIN (
    SELECT 
        category,
        AVG(price) AS avg_price
    FROM products 
    GROUP BY category
) cat_avg ON p.category = cat_avg.category
WHERE p.price > cat_avg.avg_price;

-- 복잡한 집계
SELECT 
    category,
    product_count,
    total_value
FROM (
    SELECT 
        category,
        COUNT(*) AS product_count,
        SUM(price * stock) AS total_value
    FROM products 
    GROUP BY category
) AS category_stats
WHERE total_value > 10000
ORDER BY total_value DESC;
```

### 상관 서브쿼리

```sql
-- 각 카테고리 내에서 평균보다 비싼 상품
SELECT 
    p1.product_name,
    p1.category,
    p1.price
FROM products p1
WHERE p1.price > (
    SELECT AVG(p2.price)
    FROM products p2
    WHERE p2.category = p1.category
);

-- 각 고객의 최근 주문 날짜
SELECT 
    c.customer_name,
    (SELECT MAX(order_date) 
     FROM orders o 
     WHERE o.customer_id = c.customer_id) AS last_order
FROM customers c;
```

### ANY / ALL 연산자

```sql
-- ANY: 하나라도 만족
SELECT * FROM products 
WHERE price > ANY (
    SELECT price 
    FROM products 
    WHERE category = 'Electronics'
);

-- ALL: 모두 만족
SELECT * FROM products 
WHERE price > ALL (
    SELECT price 
    FROM products 
    WHERE category = 'Books'
);

-- = ANY는 IN과 동일
SELECT * FROM products 
WHERE category = ANY (ARRAY['Electronics', 'Books']);
-- WHERE category IN ('Electronics', 'Books') 와 동일
```

### 다중 열 서브쿼리

```sql
-- 여러 열 동시 비교
SELECT * FROM products 
WHERE (category, price) IN (
    SELECT category, MAX(price)
    FROM products 
    GROUP BY category
);
```

### 실전 예제

```sql
-- 예제 1: 카테고리별 Top 3 상품
SELECT 
    category,
    product_name,
    price
FROM (
    SELECT 
        category,
        product_name,
        price,
        ROW_NUMBER() OVER (
            PARTITION BY category 
            ORDER BY price DESC
        ) AS rank
    FROM products
) ranked
WHERE rank <= 3
ORDER BY category, rank;

-- 예제 2: 평균 이상 주문한 고객
SELECT 
    customer_id,
    total_orders,
    total_amount
FROM (
    SELECT 
        customer_id,
        COUNT(*) AS total_orders,
        SUM(total_amount) AS total_amount
    FROM orders 
    GROUP BY customer_id
) customer_stats
WHERE total_amount > (
    SELECT AVG(total_amount) 
    FROM (
        SELECT SUM(total_amount) AS total_amount
        FROM orders 
        GROUP BY customer_id
    ) AS avg_calc
)
ORDER BY total_amount DESC;

-- 예제 3: 재고가 카테고리 평균보다 적은 상품
SELECT 
    p.product_name,
    p.category,
    p.stock,
    cat_avg.avg_stock
FROM products p
JOIN (
    SELECT 
        category,
        AVG(stock) AS avg_stock
    FROM products 
    GROUP BY category
) cat_avg ON p.category = cat_avg.category
WHERE p.stock < cat_avg.avg_stock
ORDER BY p.category, p.stock;
```

---

## 종합 실습

### 실습 1: 상품 분석

```sql
-- 1. 카테고리별 통계
SELECT 
    category,
    COUNT(*) AS products,
    AVG(price)::NUMERIC(10,2) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    SUM(stock) AS total_stock
FROM products 
GROUP BY category 
ORDER BY products DESC;

-- 2. 평균보다 비싼 상품
SELECT 
    product_name,
    category,
    price,
    (SELECT AVG(price) FROM products) AS avg_price
FROM products 
WHERE price > (SELECT AVG(price) FROM products)
ORDER BY price DESC;

-- 3. 카테고리별 상위 5개 상품
SELECT * FROM (
    SELECT 
        category,
        product_name,
        price,
        RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rank
    FROM products
) ranked
WHERE rank <= 5;
```

### 실습 2: 주문 분석

```sql
-- 1. 월별 주문 추이
SELECT 
    TO_CHAR(order_date, 'YYYY-MM') AS month,
    COUNT(*) AS orders,
    SUM(total_amount)::NUMERIC(12,2) AS revenue,
    AVG(total_amount)::NUMERIC(10,2) AS avg_order
FROM orders 
WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY TO_CHAR(order_date, 'YYYY-MM')
ORDER BY month;

-- 2. 고객별 최근 주문
SELECT 
    customer_id,
    MAX(order_date) AS last_order,
    COUNT(*) AS total_orders,
    SUM(total_amount) AS total_spent
FROM orders 
GROUP BY customer_id 
HAVING COUNT(*) > 1
ORDER BY last_order DESC;

-- 3. 인기 상품 Top 10
SELECT 
    p.product_name,
    COUNT(oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name 
ORDER BY total_quantity DESC 
LIMIT 10;
```

---

## 다음 단계

4단계를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ 복잡한 WHERE 조건 작성
✅ 데이터 정렬 및 페이징
✅ 집계 함수 활용
✅ GROUP BY와 HAVING으로 그룹 분석
✅ 서브쿼리 작성

**다음 학습**: 5단계 - 관계형 데이터베이스
- 조인 (INNER, LEFT, RIGHT, FULL, CROSS JOIN)
- SELF JOIN
- 제약조건 (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK, NOT NULL)
