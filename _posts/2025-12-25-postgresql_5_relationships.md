---
layout: post
title: PostgreSQL 완전 정복 5단계 - 관계형 데이터베이스 설계
summary: 모든 JOIN 타입, SELF JOIN, 외래키 제약조건까지 관계형 DB 완벽 이해
author: keonhee
date: 2025-12-25 09:00:00 +0900
category: Database
keywords: PostgreSQL, JOIN, Foreign Key, Constraints, Relational Database
permalink: /blog/postgresql_relationships/
usemathjax: false
thumbnail: /assets/img/posts/postgresql_join.png
imageNameKey: postgresql
---


# PostgreSQL 5단계: 관계형 데이터베이스

## 목차
1. [조인 (INNER, LEFT, RIGHT, FULL, CROSS JOIN)](#1-조인-inner-left-right-full-cross-join)
2. [SELF JOIN](#2-self-join)
3. [제약조건](#3-제약조건)

---

## 1. 조인 (INNER, LEFT, RIGHT, FULL, CROSS JOIN)

### 테이블 준비

```sql
-- 고객 테이블
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    city VARCHAR(50)
);

-- 주문 테이블
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE DEFAULT CURRENT_DATE,
    total_amount DECIMAL(10, 2)
);

-- 상품 테이블
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(50)
);

-- 주문 상세 테이블
CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);

-- 샘플 데이터 삽입
INSERT INTO customers (customer_name, email, city) VALUES
    ('John Doe', 'john@example.com', 'Seoul'),
    ('Jane Smith', 'jane@example.com', 'Busan'),
    ('Bob Wilson', 'bob@example.com', 'Seoul'),
    ('Alice Brown', 'alice@example.com', 'Incheon');

INSERT INTO products (product_name, price, category) VALUES
    ('Laptop', 1200.00, 'Electronics'),
    ('Mouse', 25.00, 'Electronics'),
    ('Keyboard', 75.00, 'Electronics'),
    ('Monitor', 300.00, 'Electronics'),
    ('Book', 15.00, 'Books');

INSERT INTO orders (customer_id, order_date, total_amount) VALUES
    (1, '2024-01-15', 1500.00),
    (1, '2024-02-20', 300.00),
    (2, '2024-01-25', 100.00),
    (3, '2024-03-01', 1200.00);

INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
    (1, 1, 1, 1200.00),
    (1, 2, 2, 25.00),
    (2, 4, 1, 300.00),
    (3, 2, 4, 25.00),
    (4, 1, 1, 1200.00);
```

### INNER JOIN - 양쪽에 모두 있는 데이터

```sql
-- 기본 INNER JOIN
SELECT 
    customers.customer_name,
    orders.order_id,
    orders.total_amount
FROM customers
INNER JOIN orders ON customers.customer_id = orders.customer_id;

-- 테이블 별칭 사용 (권장)
SELECT 
    c.customer_name,
    o.order_id,
    o.total_amount,
    o.order_date
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;

-- JOIN 키워드만 사용 (INNER 생략 가능)
SELECT 
    c.customer_name,
    o.order_id
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;

-- 여러 조건으로 조인
SELECT 
    c.customer_name,
    o.order_id
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id 
             AND o.total_amount > 500;
```

### 다중 테이블 조인

```sql
-- 3개 테이블 조인
SELECT 
    c.customer_name,
    o.order_id,
    p.product_name,
    oi.quantity,
    oi.price
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id;

-- WHERE 절 추가
SELECT 
    c.customer_name,
    o.order_date,
    p.product_name,
    oi.quantity,
    oi.quantity * oi.price AS item_total
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE c.city = 'Seoul'
  AND o.order_date >= '2024-01-01'
ORDER BY o.order_date DESC;
```

### LEFT JOIN (LEFT OUTER JOIN) - 왼쪽 테이블의 모든 데이터

```sql
-- 모든 고객과 그들의 주문 (주문이 없어도 고객은 표시)
SELECT 
    c.customer_name,
    c.email,
    o.order_id,
    o.total_amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;

-- 주문하지 않은 고객 찾기
SELECT 
    c.customer_name,
    c.email
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- 집계와 함께 사용
SELECT 
    c.customer_name,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name
ORDER BY total_spent DESC;
```

### RIGHT JOIN (RIGHT OUTER JOIN) - 오른쪽 테이블의 모든 데이터

```sql
-- 모든 주문과 고객 정보 (고객 정보가 없는 주문도 표시)
SELECT 
    c.customer_name,
    o.order_id,
    o.total_amount
FROM customers c
RIGHT JOIN orders o ON c.customer_id = o.customer_id;

-- LEFT JOIN으로 변환 가능
SELECT 
    c.customer_name,
    o.order_id,
    o.total_amount
FROM orders o
LEFT JOIN customers c ON c.customer_id = o.customer_id;
-- 위 두 쿼리는 동일한 결과
```

### FULL OUTER JOIN - 양쪽 테이블의 모든 데이터

```sql
-- 모든 고객과 모든 주문
SELECT 
    c.customer_name,
    c.email,
    o.order_id,
    o.total_amount
FROM customers c
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id;

-- 매칭되지 않은 데이터 찾기
SELECT 
    c.customer_name,
    o.order_id
FROM customers c
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id IS NULL 
   OR o.order_id IS NULL;
```

### CROSS JOIN - 카티션 곱

```sql
-- 모든 조합 생성
SELECT 
    c.customer_name,
    p.product_name
FROM customers c
CROSS JOIN products p;

-- 쉼표로도 가능 (구식 문법)
SELECT 
    c.customer_name,
    p.product_name
FROM customers c, products p;

-- 실용 예제: 날짜 범위와 카테고리 조합
SELECT 
    dates.date,
    categories.category
FROM (
    SELECT generate_series(
        '2024-01-01'::date,
        '2024-01-07'::date,
        '1 day'::interval
    )::date AS date
) dates
CROSS JOIN (
    SELECT DISTINCT category FROM products
) categories
ORDER BY date, category;
```

### NATURAL JOIN - 자동 열 매칭

```sql
-- 같은 이름의 열로 자동 조인 (주의해서 사용)
SELECT *
FROM customers
NATURAL JOIN orders;

-- 명시적 JOIN 사용을 권장
-- NATURAL JOIN은 예상치 못한 결과를 낼 수 있음
```

### USING 절

```sql
-- 조인 열 이름이 같을 때 간단하게 표현
SELECT 
    c.customer_name,
    o.order_id
FROM customers c
JOIN orders o USING (customer_id);
-- ON c.customer_id = o.customer_id 와 동일
```

### 조인 성능 최적화

```sql
-- 1. 인덱스 사용
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- 2. 작은 테이블을 먼저 (옵티마이저가 자동으로 처리)
-- 3. 필요한 열만 선택
SELECT 
    c.customer_id,
    c.customer_name,
    o.order_id
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;
-- SELECT * 보다 효율적

-- 4. WHERE 절로 먼저 필터링
SELECT 
    c.customer_name,
    o.order_id
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.city = 'Seoul'  -- 먼저 필터링
  AND o.order_date >= '2024-01-01';
```

### 실전 예제

```sql
-- 예제 1: 고객별 주문 요약
SELECT 
    c.customer_id,
    c.customer_name,
    c.city,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS total_spent,
    MAX(o.order_date) AS last_order_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name, c.city
ORDER BY total_spent DESC;

-- 예제 2: 상품별 판매 통계
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    COUNT(oi.item_id) AS times_sold,
    COALESCE(SUM(oi.quantity), 0) AS total_quantity,
    COALESCE(SUM(oi.quantity * oi.price), 0) AS total_revenue
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC;

-- 예제 3: 최근 주문 상세
SELECT 
    o.order_id,
    o.order_date,
    c.customer_name,
    c.email,
    p.product_name,
    oi.quantity,
    oi.price,
    oi.quantity * oi.price AS item_total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY o.order_date DESC, o.order_id;
```

---

## 2. SELF JOIN

### 기본 개념

```sql
-- 직원 테이블 (계층 구조)
CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    emp_name VARCHAR(100) NOT NULL,
    manager_id INTEGER REFERENCES employees(emp_id),
    position VARCHAR(50),
    salary DECIMAL(10, 2)
);

INSERT INTO employees (emp_name, manager_id, position, salary) VALUES
    ('CEO Kim', NULL, 'CEO', 150000),
    ('Manager Park', 1, 'Manager', 100000),
    ('Manager Lee', 1, 'Manager', 100000),
    ('Developer Choi', 2, 'Developer', 70000),
    ('Developer Jung', 2, 'Developer', 70000),
    ('Designer Han', 3, 'Designer', 65000);
```

### 직원과 상사 조회

```sql
-- 각 직원과 그들의 상사
SELECT 
    e.emp_name AS employee,
    e.position AS employee_position,
    m.emp_name AS manager,
    m.position AS manager_position
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

-- 상사가 있는 직원만
SELECT 
    e.emp_name AS employee,
    m.emp_name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.emp_id;
```

### 계층 구조 조회

```sql
-- 레벨 표시
SELECT 
    e.emp_name,
    e.position,
    COALESCE(m.emp_name, 'No Manager') AS manager,
    CASE 
        WHEN e.manager_id IS NULL THEN 0
        WHEN m.manager_id IS NULL THEN 1
        ELSE 2
    END AS level
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id
ORDER BY level, e.emp_name;

-- 재귀 CTE로 전체 계층 조회
WITH RECURSIVE emp_hierarchy AS (
    -- 최상위 (CEO)
    SELECT 
        emp_id,
        emp_name,
        manager_id,
        position,
        0 AS level,
        emp_name::TEXT AS path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- 하위 직원들
    SELECT 
        e.emp_id,
        e.emp_name,
        e.manager_id,
        e.position,
        eh.level + 1,
        eh.path || ' > ' || e.emp_name
    FROM employees e
    JOIN emp_hierarchy eh ON e.manager_id = eh.emp_id
)
SELECT 
    REPEAT('  ', level) || emp_name AS org_chart,
    position,
    level
FROM emp_hierarchy
ORDER BY path;
```

### 동일 조건 찾기

```sql
-- 같은 매니저를 가진 동료 찾기
SELECT 
    e1.emp_name AS employee1,
    e2.emp_name AS employee2,
    m.emp_name AS shared_manager
FROM employees e1
JOIN employees e2 ON e1.manager_id = e2.manager_id 
                 AND e1.emp_id < e2.emp_id  -- 중복 방지
JOIN employees m ON e1.manager_id = m.emp_id;

-- 같은 급여를 받는 직원
SELECT 
    e1.emp_name AS employee1,
    e2.emp_name AS employee2,
    e1.salary
FROM employees e1
JOIN employees e2 ON e1.salary = e2.salary 
                 AND e1.emp_id < e2.emp_id;
```

### 비교 분석

```sql
-- 평균보다 높은 급여를 받는 직원
SELECT 
    e.emp_name,
    e.salary,
    AVG(e2.salary) AS avg_salary
FROM employees e
CROSS JOIN employees e2
GROUP BY e.emp_id, e.emp_name, e.salary
HAVING e.salary > AVG(e2.salary);
```

---

## 3. 제약조건

### PRIMARY KEY (기본 키)

```sql
-- 생성 시 지정
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 열 제약으로 지정
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name VARCHAR(100)
);

-- 테이블 제약으로 지정
CREATE TABLE students (
    student_id INTEGER,
    name VARCHAR(100),
    PRIMARY KEY (student_id)
);

-- 복합 기본 키
CREATE TABLE course_enrollment (
    student_id INTEGER,
    course_id INTEGER,
    enrollment_date DATE,
    PRIMARY KEY (student_id, course_id)
);

-- 기존 테이블에 추가
ALTER TABLE students ADD PRIMARY KEY (student_id);

-- 기본 키 삭제
ALTER TABLE students DROP CONSTRAINT students_pkey;
```

### FOREIGN KEY (외래 키)

```sql
-- 기본 외래 키
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id)
);

-- 테이블 제약으로 지정
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 이름 지정
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    CONSTRAINT fk_customer 
        FOREIGN KEY (customer_id) 
        REFERENCES customers(customer_id)
);
```

### 외래 키 옵션

```sql
-- CASCADE: 참조된 행 삭제 시 함께 삭제
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- SET NULL: 참조된 행 삭제 시 NULL로 설정
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id)
        ON DELETE SET NULL
);

-- SET DEFAULT: 기본값으로 설정
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER DEFAULT 1 
        REFERENCES customers(customer_id)
        ON DELETE SET DEFAULT
);

-- RESTRICT: 참조된 행이 있으면 삭제 방지 (기본값)
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id)
        ON DELETE RESTRICT
);

-- NO ACTION: RESTRICT와 유사하지만 체크 시점이 다름
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id)
        ON DELETE NO ACTION
);
```

### UNIQUE (고유 제약)

```sql
-- 열 제약
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(100) UNIQUE,
    username VARCHAR(50) UNIQUE
);

-- 테이블 제약
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(100),
    username VARCHAR(50),
    UNIQUE (email),
    UNIQUE (username)
);

-- 이름 지정
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(100),
    CONSTRAINT unique_email UNIQUE (email)
);

-- 복합 고유 제약
CREATE TABLE product_reviews (
    review_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    product_id INTEGER,
    review_text TEXT,
    UNIQUE (user_id, product_id)  -- 같은 사용자는 같은 상품에 한 번만 리뷰
);

-- 기존 테이블에 추가
ALTER TABLE users ADD UNIQUE (email);
ALTER TABLE users ADD CONSTRAINT unique_username UNIQUE (username);

-- 삭제
ALTER TABLE users DROP CONSTRAINT unique_email;
```

### CHECK (체크 제약)

```sql
-- 단순 체크
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    price DECIMAL(10, 2) CHECK (price > 0),
    stock INTEGER CHECK (stock >= 0)
);

-- 이름 지정
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    price DECIMAL(10, 2),
    stock INTEGER,
    CONSTRAINT check_price_positive CHECK (price > 0),
    CONSTRAINT check_stock_non_negative CHECK (stock >= 0)
);

-- 복잡한 체크
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    price DECIMAL(10, 2),
    discount_price DECIMAL(10, 2),
    CHECK (discount_price < price)
);

-- 여러 열 체크
CREATE TABLE events (
    event_id SERIAL PRIMARY KEY,
    start_date DATE,
    end_date DATE,
    CHECK (end_date >= start_date)
);

-- 범위 체크
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    age INTEGER CHECK (age BETWEEN 18 AND 100),
    grade CHAR(1) CHECK (grade IN ('A', 'B', 'C', 'D', 'F'))
);

-- 기존 테이블에 추가
ALTER TABLE products 
ADD CONSTRAINT check_price CHECK (price > 0);

-- 삭제
ALTER TABLE products DROP CONSTRAINT check_price;
```

### NOT NULL (NOT NULL 제약)

```sql
-- 생성 시 지정
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    phone VARCHAR(20)  -- NULL 허용
);

-- 기존 테이블에 추가
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- 제거
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

-- NULL 값이 있는 경우 먼저 처리
UPDATE users SET email = 'unknown@example.com' WHERE email IS NULL;
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
```

### DEFAULT (기본값)

```sql
-- 생성 시 지정
CREATE TABLE posts (
    post_id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'draft',
    view_count INTEGER DEFAULT 0,
    is_published BOOLEAN DEFAULT false
);

-- 함수 사용
CREATE TABLE logs (
    log_id SERIAL PRIMARY KEY,
    log_message TEXT,
    log_time TIMESTAMP DEFAULT NOW(),
    log_date DATE DEFAULT CURRENT_DATE
);

-- 기존 테이블에 추가
ALTER TABLE posts ALTER COLUMN status SET DEFAULT 'draft';

-- 제거
ALTER TABLE posts ALTER COLUMN status DROP DEFAULT;

-- DEFAULT 키워드로 삽입
INSERT INTO posts (title, status) VALUES ('Test', DEFAULT);
INSERT INTO posts (title) VALUES ('Test');  -- 자동으로 DEFAULT 사용
```

### 제약조건 확인

```sql
-- 테이블의 모든 제약조건 확인
SELECT 
    conname AS constraint_name,
    contype AS constraint_type,
    pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid = 'products'::regclass;

-- 제약조건 타입:
-- p = PRIMARY KEY
-- f = FOREIGN KEY  
-- u = UNIQUE
-- c = CHECK
-- t = TRIGGER
-- x = EXCLUSION

-- 외래 키 관계 확인
SELECT
    tc.table_name AS child_table,
    kcu.column_name AS child_column,
    ccu.table_name AS parent_table,
    ccu.column_name AS parent_column,
    rc.delete_rule,
    rc.update_rule
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
  ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.referential_constraints rc
  ON tc.constraint_name = rc.constraint_name
JOIN information_schema.constraint_column_usage ccu
  ON rc.unique_constraint_name = ccu.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
ORDER BY tc.table_name;
```

### 실전 예제

```sql
-- 완전한 이커머스 스키마
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_code VARCHAR(20) NOT NULL UNIQUE,
    product_name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_price_positive CHECK (price > 0),
    CONSTRAINT check_stock_non_negative CHECK (stock >= 0)
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10, 2) NOT NULL,
    CONSTRAINT fk_customer 
        FOREIGN KEY (customer_id) 
        REFERENCES customers(customer_id)
        ON DELETE RESTRICT,
    CONSTRAINT check_total_positive CHECK (total_amount > 0),
    CONSTRAINT check_status CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    CONSTRAINT fk_order 
        FOREIGN KEY (order_id) 
        REFERENCES orders(order_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_product 
        FOREIGN KEY (product_id) 
        REFERENCES products(product_id)
        ON DELETE RESTRICT,
    CONSTRAINT check_quantity_positive CHECK (quantity > 0),
    CONSTRAINT check_price_positive CHECK (price > 0),
    UNIQUE (order_id, product_id)
);
```

---

## 종합 실습

```sql
-- 실습: 도서관 관리 시스템

-- 1. 회원 테이블
CREATE TABLE members (
    member_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(20),
    join_date DATE DEFAULT CURRENT_DATE,
    membership_type VARCHAR(20) DEFAULT 'basic',
    CONSTRAINT check_membership CHECK (membership_type IN ('basic', 'premium'))
);

-- 2. 도서 테이블
CREATE TABLE books (
    book_id SERIAL PRIMARY KEY,
    isbn VARCHAR(13) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    author VARCHAR(100) NOT NULL,
    publisher VARCHAR(100),
    published_year INTEGER,
    total_copies INTEGER DEFAULT 1,
    available_copies INTEGER DEFAULT 1,
    CONSTRAINT check_copies CHECK (available_copies <= total_copies),
    CONSTRAINT check_total_positive CHECK (total_copies > 0)
);

-- 3. 대출 테이블
CREATE TABLE loans (
    loan_id SERIAL PRIMARY KEY,
    member_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    loan_date DATE DEFAULT CURRENT_DATE,
    due_date DATE NOT NULL,
    return_date DATE,
    CONSTRAINT fk_member 
        FOREIGN KEY (member_id) 
        REFERENCES members(member_id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_book 
        FOREIGN KEY (book_id) 
        REFERENCES books(book_id)
        ON DELETE RESTRICT,
    CONSTRAINT check_dates CHECK (due_date > loan_date)
);

-- 4. 복잡한 조회
SELECT 
    m.name,
    m.membership_type,
    COUNT(l.loan_id) AS total_loans,
    COUNT(CASE WHEN l.return_date IS NULL THEN 1 END) AS current_loans
FROM members m
LEFT JOIN loans l ON m.member_id = l.member_id
GROUP BY m.member_id, m.name, m.membership_type
ORDER BY total_loans DESC;
```

---

## 다음 단계

5단계를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ 다양한 조인 활용 (INNER, LEFT, RIGHT, FULL, CROSS)
✅ SELF JOIN으로 계층 구조 처리
✅ 제약조건으로 데이터 무결성 보장

**다음 학습**: 6단계 - 성능 최적화
- 인덱스 생성 및 관리
- 쿼리 실행 계획 (EXPLAIN)
- 성능 튜닝 기법
