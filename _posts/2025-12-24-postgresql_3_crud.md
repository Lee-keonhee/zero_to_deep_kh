---
layout: post
title: PostgreSQL 완전 정복 3단계 - 데이터 조작 마스터하기
summary: INSERT, SELECT, UPDATE, DELETE부터 UPSERT까지 모든 데이터 조작 기법
author: keonhee
date: 2025-12-24 09:00:00 +0900
category: Database
keywords: PostgreSQL, CRUD, INSERT, SELECT, UPDATE, DELETE, UPSERT
permalink: /blog/postgresql_crud/
usemathjax: false
thumbnail: /assets/img/posts/postgresql.png
imageNameKey: postgresql
---


## 목차
1. [INSERT - 데이터 삽입](#1-insert---데이터-삽입)
2. [SELECT - 데이터 조회 기초](#2-select---데이터-조회-기초)
3. [UPDATE - 데이터 수정](#3-update---데이터-수정)
4. [DELETE - 데이터 삭제](#4-delete---데이터-삭제)

---

## 1. INSERT - 데이터 삽입

### 기본 INSERT 문법

```sql
-- 모든 열에 값 삽입
INSERT INTO table_name VALUES (value1, value2, value3);

-- 특정 열에만 값 삽입
INSERT INTO table_name (column1, column2) VALUES (value1, value2);
```

### 단일 행 삽입

```sql
-- 테이블 생성
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INTEGER,
    enrollment_date DATE DEFAULT CURRENT_DATE
);

-- 모든 열 지정하여 삽입
INSERT INTO students (name, email, age, enrollment_date)
VALUES ('John Doe', 'john@example.com', 20, '2024-01-15');

-- 일부 열만 지정 (나머지는 기본값 또는 NULL)
INSERT INTO students (name, email)
VALUES ('Jane Smith', 'jane@example.com');

-- DEFAULT 키워드 사용
INSERT INTO students (name, email, age, enrollment_date)
VALUES ('Bob Wilson', 'bob@example.com', 22, DEFAULT);

-- 순서대로 모든 값 삽입 (열 이름 생략)
-- 주의: 구조 변경 시 오류 발생 가능
INSERT INTO students 
VALUES (DEFAULT, 'Alice Brown', 'alice@example.com', 21, DEFAULT);
```

### 여러 행 동시 삽입

```sql
-- 한 번의 쿼리로 여러 행 삽입 (권장)
INSERT INTO students (name, email, age) VALUES
    ('Charlie Davis', 'charlie@example.com', 23),
    ('Diana Evans', 'diana@example.com', 19),
    ('Edward Frank', 'edward@example.com', 24),
    ('Fiona Green', 'fiona@example.com', 20);

-- 성능 비교: 개별 삽입 vs 일괄 삽입
-- 느림 (여러 번 실행)
INSERT INTO students (name, email) VALUES ('Test1', 'test1@example.com');
INSERT INTO students (name, email) VALUES ('Test2', 'test2@example.com');
INSERT INTO students (name, email) VALUES ('Test3', 'test3@example.com');

-- 빠름 (한 번 실행)
INSERT INTO students (name, email) VALUES
    ('Test1', 'test1@example.com'),
    ('Test2', 'test2@example.com'),
    ('Test3', 'test3@example.com');
```

### RETURNING 절 - 삽입된 데이터 반환

```sql
-- 삽입된 행의 모든 열 반환
INSERT INTO students (name, email, age)
VALUES ('George Harris', 'george@example.com', 21)
RETURNING *;

-- 특정 열만 반환
INSERT INTO students (name, email, age)
VALUES ('Helen Johnson', 'helen@example.com', 22)
RETURNING student_id, name;

-- 여러 행 삽입 후 모두 반환
INSERT INTO students (name, email, age) VALUES
    ('Ivan King', 'ivan@example.com', 20),
    ('Julia Lee', 'julia@example.com', 23)
RETURNING student_id, name, enrollment_date;

-- 계산된 값 반환
INSERT INTO students (name, email, age)
VALUES ('Kevin Miller', 'kevin@example.com', 19)
RETURNING student_id, name, age * 12 AS age_in_months;
```

### INSERT ... SELECT (서브쿼리로 삽입)

```sql
-- 다른 테이블에서 데이터 복사
CREATE TABLE students_archive (
    student_id INTEGER,
    name VARCHAR(100),
    email VARCHAR(100),
    age INTEGER,
    archived_date DATE DEFAULT CURRENT_DATE
);

-- SELECT 결과를 새 테이블에 삽입
INSERT INTO students_archive (student_id, name, email, age)
SELECT student_id, name, email, age
FROM students
WHERE age > 22;

-- 조건부 복사
INSERT INTO students_archive (student_id, name, email, age)
SELECT student_id, name, email, age
FROM students
WHERE enrollment_date < '2024-01-01';
```

### ON CONFLICT (중복 처리)

```sql
-- 중복 시 무시 (IGNORE)
INSERT INTO students (name, email, age)
VALUES ('John Doe', 'john@example.com', 25)
ON CONFLICT (email) DO NOTHING;

-- 중복 시 업데이트 (UPSERT)
INSERT INTO students (name, email, age)
VALUES ('John Doe', 'john@example.com', 25)
ON CONFLICT (email) 
DO UPDATE SET 
    name = EXCLUDED.name,
    age = EXCLUDED.age;

-- 여러 열 업데이트
INSERT INTO students (name, email, age)
VALUES ('John Doe', 'john@example.com', 25)
ON CONFLICT (email)
DO UPDATE SET 
    name = EXCLUDED.name,
    age = EXCLUDED.age,
    enrollment_date = CURRENT_DATE;

-- 조건부 업데이트
INSERT INTO students (name, email, age)
VALUES ('John Doe', 'john@example.com', 25)
ON CONFLICT (email)
DO UPDATE SET age = EXCLUDED.age
WHERE students.age < EXCLUDED.age;  -- age가 더 클 때만 업데이트

-- RETURNING과 함께 사용
INSERT INTO students (name, email, age)
VALUES ('John Doe', 'john@example.com', 25)
ON CONFLICT (email)
DO UPDATE SET age = EXCLUDED.age
RETURNING *, xmax = 0 AS inserted;  -- true면 삽입, false면 업데이트
```

### 실전 예제

```sql
-- 상품 테이블
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_code VARCHAR(20) UNIQUE NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock INTEGER DEFAULT 0,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 상품 데이터 삽입
INSERT INTO products (product_code, product_name, price, stock, category) VALUES
    ('LAPTOP001', 'Dell XPS 13', 1299.99, 10, 'Electronics'),
    ('LAPTOP002', 'MacBook Pro', 2399.99, 5, 'Electronics'),
    ('PHONE001', 'iPhone 15', 999.99, 20, 'Electronics'),
    ('BOOK001', 'PostgreSQL Guide', 49.99, 100, 'Books'),
    ('BOOK002', 'Python for Beginners', 39.99, 50, 'Books');

-- UPSERT 예제: 상품 코드가 존재하면 재고만 업데이트
INSERT INTO products (product_code, product_name, price, stock, category)
VALUES ('LAPTOP001', 'Dell XPS 13', 1299.99, 15, 'Electronics')
ON CONFLICT (product_code)
DO UPDATE SET 
    stock = products.stock + EXCLUDED.stock,
    updated_at = NOW()
RETURNING *;
```

---

## 2. SELECT - 데이터 조회 기초

### 기본 SELECT 문법

```sql
-- 모든 열 조회
SELECT * FROM students;

-- 특정 열만 조회
SELECT name, email FROM students;

-- 열 이름 변경 (AS 별칭)
SELECT 
    name AS student_name,
    email AS contact_email,
    age AS current_age
FROM students;
```

### WHERE 절 - 조건 필터링

```sql
-- 단일 조건
SELECT * FROM students WHERE age = 20;

-- 비교 연산자
SELECT * FROM students WHERE age > 20;
SELECT * FROM students WHERE age >= 21;
SELECT * FROM students WHERE age < 23;
SELECT * FROM students WHERE age <= 22;
SELECT * FROM students WHERE age != 20;
SELECT * FROM students WHERE age <> 20;  -- != 와 동일

-- 문자열 비교
SELECT * FROM students WHERE name = 'John Doe';

-- LIKE 패턴 매칭
SELECT * FROM students WHERE name LIKE 'J%';      -- J로 시작
SELECT * FROM students WHERE name LIKE '%son';    -- son으로 끝남
SELECT * FROM students WHERE name LIKE '%a%';     -- a 포함
SELECT * FROM students WHERE name LIKE '_oh%';    -- 두 번째 글자가 oh

-- ILIKE (대소문자 구분 없음)
SELECT * FROM students WHERE email ILIKE '%EXAMPLE.COM';

-- IN 연산자
SELECT * FROM students WHERE age IN (20, 21, 22);
SELECT * FROM students WHERE name IN ('John Doe', 'Jane Smith');

-- BETWEEN 연산자
SELECT * FROM students WHERE age BETWEEN 20 AND 23;
SELECT * FROM students 
WHERE enrollment_date BETWEEN '2024-01-01' AND '2024-12-31';

-- NULL 체크
SELECT * FROM students WHERE email IS NULL;
SELECT * FROM students WHERE email IS NOT NULL;
```

### 논리 연산자

```sql
-- AND 연산자
SELECT * FROM students 
WHERE age > 20 AND email LIKE '%example.com';

-- OR 연산자
SELECT * FROM students 
WHERE age < 20 OR age > 23;

-- NOT 연산자
SELECT * FROM students 
WHERE NOT age = 20;

-- 복합 조건 (괄호 사용)
SELECT * FROM students 
WHERE (age > 20 AND email LIKE '%example.com')
   OR (age < 19 AND name LIKE 'J%');
```

### ORDER BY - 정렬

```sql
-- 오름차순 정렬 (기본값)
SELECT * FROM students ORDER BY age;
SELECT * FROM students ORDER BY age ASC;

-- 내림차순 정렬
SELECT * FROM students ORDER BY age DESC;

-- 여러 열로 정렬
SELECT * FROM students 
ORDER BY age DESC, name ASC;

-- NULL 값 정렬
SELECT * FROM students ORDER BY email NULLS FIRST;
SELECT * FROM students ORDER BY email NULLS LAST;

-- 계산 결과로 정렬
SELECT name, age, age * 12 AS age_in_months
FROM students
ORDER BY age_in_months DESC;

-- 열 위치로 정렬 (비권장)
SELECT name, age FROM students ORDER BY 2 DESC;  -- 2번째 열(age)
```

### LIMIT와 OFFSET

```sql
-- 상위 N개 행만 조회
SELECT * FROM students LIMIT 5;

-- N번째부터 조회 (페이징)
SELECT * FROM students LIMIT 5 OFFSET 10;  -- 11~15번째 행

-- 페이징 예제
-- 1페이지 (1~10)
SELECT * FROM students ORDER BY student_id LIMIT 10 OFFSET 0;

-- 2페이지 (11~20)
SELECT * FROM students ORDER BY student_id LIMIT 10 OFFSET 10;

-- 3페이지 (21~30)
SELECT * FROM students ORDER BY student_id LIMIT 10 OFFSET 20;

-- 최신 10개 데이터
SELECT * FROM students 
ORDER BY enrollment_date DESC 
LIMIT 10;
```

### DISTINCT - 중복 제거

```sql
-- 중복 제거
SELECT DISTINCT age FROM students;

-- 여러 열 조합의 중복 제거
SELECT DISTINCT age, category FROM students;

-- COUNT와 함께 사용
SELECT COUNT(DISTINCT age) FROM students;

-- DISTINCT ON (PostgreSQL 특화)
SELECT DISTINCT ON (age) name, age, email
FROM students
ORDER BY age, name;  -- age별로 첫 번째 행만
```

### 계산과 표현식

```sql
-- 산술 연산
SELECT 
    name,
    age,
    age + 10 AS age_plus_ten,
    age * 12 AS age_in_months,
    age / 2.0 AS half_age
FROM students;

-- 문자열 연결
SELECT 
    name || ' (' || email || ')' AS full_info
FROM students;

-- CONCAT 함수
SELECT 
    CONCAT(name, ' - ', email) AS student_info
FROM students;

-- CASE 문
SELECT 
    name,
    age,
    CASE 
        WHEN age < 20 THEN 'Teenager'
        WHEN age BETWEEN 20 AND 22 THEN 'Young Adult'
        WHEN age > 22 THEN 'Adult'
        ELSE 'Unknown'
    END AS age_group
FROM students;

-- COALESCE (NULL 대체)
SELECT 
    name,
    COALESCE(email, 'No email') AS email
FROM students;
```

### 집계 함수

```sql
-- COUNT - 개수 세기
SELECT COUNT(*) FROM students;
SELECT COUNT(email) FROM students;  -- NULL 제외
SELECT COUNT(DISTINCT age) FROM students;

-- SUM - 합계
SELECT SUM(age) FROM students;

-- AVG - 평균
SELECT AVG(age) FROM students;
SELECT AVG(age)::NUMERIC(10,2) FROM students;  -- 소수점 2자리

-- MAX, MIN - 최대값, 최소값
SELECT MAX(age) FROM students;
SELECT MIN(age) FROM students;

-- 여러 집계 함수 동시 사용
SELECT 
    COUNT(*) AS total_students,
    AVG(age) AS average_age,
    MIN(age) AS youngest,
    MAX(age) AS oldest
FROM students;
```

### 실전 예제

```sql
-- 예제 1: 상품 조회
SELECT 
    product_code,
    product_name,
    price,
    stock,
    price * stock AS total_value
FROM products
WHERE category = 'Electronics'
  AND stock > 0
ORDER BY price DESC
LIMIT 10;

-- 예제 2: 조건별 분류
SELECT 
    product_name,
    price,
    CASE 
        WHEN price < 100 THEN 'Budget'
        WHEN price BETWEEN 100 AND 500 THEN 'Mid-range'
        ELSE 'Premium'
    END AS price_category,
    CASE 
        WHEN stock = 0 THEN 'Out of Stock'
        WHEN stock < 10 THEN 'Low Stock'
        ELSE 'In Stock'
    END AS stock_status
FROM products
ORDER BY price;

-- 예제 3: 검색 기능
SELECT * FROM products
WHERE product_name ILIKE '%laptop%'
   OR category ILIKE '%laptop%'
ORDER BY price ASC;
```

---

## 3. UPDATE - 데이터 수정

### 기본 UPDATE 문법

```sql
-- 기본 구조
UPDATE table_name
SET column1 = value1, column2 = value2
WHERE condition;
```

### 단일 열 수정

```sql
-- 특정 학생의 나이 수정
UPDATE students
SET age = 21
WHERE student_id = 1;

-- 이메일 수정
UPDATE students
SET email = 'newemail@example.com'
WHERE name = 'John Doe';

-- 조건 없이 모든 행 수정 (주의!)
UPDATE students
SET enrollment_date = CURRENT_DATE;
```

### 여러 열 동시 수정

```sql
-- 여러 열 한 번에 수정
UPDATE students
SET 
    email = 'updated@example.com',
    age = 23
WHERE student_id = 2;

-- 계산 결과로 수정
UPDATE students
SET age = age + 1
WHERE student_id = 3;
```

### 조건부 UPDATE

```sql
-- 나이가 20 미만인 학생들만 수정
UPDATE students
SET age = 20
WHERE age < 20;

-- 이메일이 없는 학생들에게 기본 이메일 설정
UPDATE students
SET email = name || '@school.edu'
WHERE email IS NULL;

-- 여러 조건
UPDATE students
SET age = age + 1
WHERE age BETWEEN 18 AND 22 
  AND enrollment_date < '2024-01-01';
```

### CASE를 사용한 조건부 UPDATE

```sql
-- 나이 그룹별로 다른 값 설정
UPDATE students
SET email = CASE 
    WHEN age < 20 THEN 'junior@school.edu'
    WHEN age BETWEEN 20 AND 22 THEN 'senior@school.edu'
    ELSE 'graduate@school.edu'
END
WHERE email IS NULL;

-- 카테고리별 가격 조정
UPDATE products
SET price = CASE 
    WHEN category = 'Electronics' THEN price * 1.1
    WHEN category = 'Books' THEN price * 1.05
    ELSE price
END;
```

### FROM 절을 사용한 UPDATE

```sql
-- 다른 테이블의 값을 참조하여 수정
CREATE TABLE student_scores (
    student_id INTEGER,
    score INTEGER
);

UPDATE students
SET age = students.age + (student_scores.score / 10)
FROM student_scores
WHERE students.student_id = student_scores.student_id;
```

### RETURNING 절

```sql
-- 수정된 데이터 반환
UPDATE students
SET age = 24
WHERE student_id = 1
RETURNING *;

-- 특정 열만 반환
UPDATE students
SET age = age + 1
WHERE age < 20
RETURNING student_id, name, age;

-- 수정 전후 비교
UPDATE students
SET age = 25
WHERE student_id = 2
RETURNING 
    student_id,
    name,
    age AS new_age,
    age - 1 AS old_age;
```

### 실전 예제

```sql
-- 예제 1: 재고 업데이트
UPDATE products
SET 
    stock = stock - 1,
    updated_at = NOW()
WHERE product_code = 'LAPTOP001'
  AND stock > 0
RETURNING product_name, stock;

-- 예제 2: 가격 할인
UPDATE products
SET 
    price = price * 0.9,  -- 10% 할인
    updated_at = NOW()
WHERE category = 'Books'
  AND stock > 50
RETURNING product_name, price;

-- 예제 3: 대량 업데이트
UPDATE products
SET stock = CASE 
    WHEN stock < 5 THEN stock + 100
    WHEN stock BETWEEN 5 AND 20 THEN stock + 50
    ELSE stock + 10
END,
updated_at = NOW()
WHERE category = 'Electronics';

-- 예제 4: 조건부 이메일 업데이트
UPDATE students
SET email = LOWER(REPLACE(name, ' ', '.')) || '@university.edu'
WHERE email IS NULL
RETURNING student_id, name, email;
```

---

## 4. DELETE - 데이터 삭제

### 기본 DELETE 문법

```sql
-- 기본 구조
DELETE FROM table_name
WHERE condition;
```

### 조건부 삭제

```sql
-- 특정 행 삭제
DELETE FROM students
WHERE student_id = 1;

-- 여러 조건
DELETE FROM students
WHERE age < 18 
  AND enrollment_date < '2020-01-01';

-- IN 연산자 사용
DELETE FROM students
WHERE student_id IN (1, 2, 3, 4, 5);

-- LIKE 패턴 사용
DELETE FROM students
WHERE email LIKE '%@temporary.com';
```

### 전체 삭제

```sql
-- 모든 데이터 삭제 (주의!)
DELETE FROM students;

-- 더 빠른 전체 삭제 (TRUNCATE)
TRUNCATE TABLE students;

-- TRUNCATE vs DELETE
-- DELETE: 느림, 롤백 가능, 트리거 실행
-- TRUNCATE: 빠름, 디스크 공간 즉시 회수, 트리거 실행 안 됨

-- CASCADE와 함께 (외래 키 참조 테이블도 삭제)
TRUNCATE TABLE students CASCADE;

-- RESTART IDENTITY (시퀀스 초기화)
TRUNCATE TABLE students RESTART IDENTITY;
```

### RETURNING 절

```sql
-- 삭제된 데이터 반환
DELETE FROM students
WHERE student_id = 1
RETURNING *;

-- 특정 열만 반환
DELETE FROM students
WHERE age > 25
RETURNING student_id, name, email;

-- 삭제된 개수 확인
WITH deleted AS (
    DELETE FROM students
    WHERE age < 18
    RETURNING *
)
SELECT COUNT(*) FROM deleted;
```

### USING 절 (다른 테이블 참조)

```sql
-- 다른 테이블의 조건으로 삭제
DELETE FROM students
USING student_scores
WHERE students.student_id = student_scores.student_id
  AND student_scores.score < 50;

-- 여러 테이블 조인
DELETE FROM students
USING enrollments, courses
WHERE students.student_id = enrollments.student_id
  AND enrollments.course_id = courses.course_id
  AND courses.status = 'cancelled';
```

### 서브쿼리를 사용한 삭제

```sql
-- 서브쿼리 결과에 해당하는 행 삭제
DELETE FROM students
WHERE student_id IN (
    SELECT student_id 
    FROM student_scores 
    WHERE score < 60
);

-- NOT IN 사용
DELETE FROM students
WHERE student_id NOT IN (
    SELECT student_id 
    FROM active_enrollments
);

-- EXISTS 사용
DELETE FROM students s
WHERE EXISTS (
    SELECT 1 
    FROM inactive_list il
    WHERE il.student_id = s.student_id
);
```

### 안전한 삭제 패턴

```sql
-- 1. 먼저 조회하여 확인
SELECT * FROM students
WHERE age < 18;

-- 2. 개수 확인
SELECT COUNT(*) FROM students
WHERE age < 18;

-- 3. 트랜잭션 사용
BEGIN;

DELETE FROM students
WHERE age < 18
RETURNING *;

-- 결과 확인 후 커밋 또는 롤백
COMMIT;  -- 또는 ROLLBACK;

-- 4. 백업 후 삭제
CREATE TABLE students_backup AS
SELECT * FROM students
WHERE age < 18;

DELETE FROM students
WHERE age < 18;
```

### 실전 예제

```sql
-- 예제 1: 오래된 데이터 삭제
DELETE FROM products
WHERE created_at < NOW() - INTERVAL '1 year'
  AND stock = 0
RETURNING product_code, product_name, created_at;

-- 예제 2: 중복 데이터 삭제 (가장 오래된 것만 남기기)
DELETE FROM students s1
USING students s2
WHERE s1.email = s2.email
  AND s1.student_id > s2.student_id;

-- 예제 3: 조건부 대량 삭제
DELETE FROM products
WHERE category = 'Discontinued'
   OR (stock = 0 AND updated_at < NOW() - INTERVAL '6 months')
RETURNING product_code, category, stock;

-- 예제 4: 아카이빙 후 삭제
BEGIN;

-- 1. 아카이브 테이블로 복사
INSERT INTO students_archive
SELECT * FROM students
WHERE enrollment_date < '2020-01-01';

-- 2. 원본에서 삭제
DELETE FROM students
WHERE enrollment_date < '2020-01-01';

COMMIT;
```

---

## 종합 실습

### 실습 1: 학생 관리 시스템

```sql
-- 1. 데이터 삽입
INSERT INTO students (name, email, age) VALUES
    ('Alice Johnson', 'alice@university.edu', 20),
    ('Bob Smith', 'bob@university.edu', 19),
    ('Charlie Brown', 'charlie@university.edu', 22),
    ('Diana Prince', 'diana@university.edu', 21);

-- 2. 데이터 조회
SELECT * FROM students ORDER BY age DESC;

SELECT name, age FROM students WHERE age >= 20;

SELECT 
    name,
    CASE 
        WHEN age < 20 THEN 'Freshman'
        WHEN age = 20 THEN 'Sophomore'
        WHEN age = 21 THEN 'Junior'
        ELSE 'Senior'
    END AS year
FROM students;

-- 3. 데이터 수정
UPDATE students
SET age = age + 1
WHERE name = 'Bob Smith'
RETURNING *;

UPDATE students
SET email = LOWER(REPLACE(name, ' ', '.')) || '@university.edu';

-- 4. 데이터 삭제
DELETE FROM students
WHERE age > 23
RETURNING *;
```

### 실습 2: 전자상거래 상품 관리

```sql
-- 1. 상품 대량 입력
INSERT INTO products (product_code, product_name, price, stock, category) VALUES
    ('ELC001', 'Wireless Mouse', 29.99, 150, 'Electronics'),
    ('ELC002', 'USB-C Cable', 12.99, 300, 'Electronics'),
    ('BOOK001', 'SQL Mastery', 45.00, 80, 'Books'),
    ('BOOK002', 'Web Development', 52.00, 60, 'Books'),
    ('GAME001', 'Chess Set', 35.99, 40, 'Games');

-- 2. 상품 검색
SELECT * FROM products 
WHERE category = 'Electronics' 
  AND price < 50
ORDER BY price ASC;

-- 3. 재고 관리
UPDATE products
SET stock = stock - 5
WHERE product_code = 'ELC001'
RETURNING product_name, stock;

UPDATE products
SET price = price * 0.85
WHERE category = 'Books'
  AND stock > 70
RETURNING product_name, price;

-- 4. 품절 상품 제거
DELETE FROM products
WHERE stock = 0
RETURNING product_code, product_name;

-- 5. 통계 조회
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price,
    SUM(stock) AS total_stock
FROM products
GROUP BY category;
```

### 실습 3: UPSERT 활용

```sql
-- 상품 재고 업데이트 (있으면 재고 추가, 없으면 새로 생성)
INSERT INTO products (product_code, product_name, price, stock, category)
VALUES ('ELC003', 'Keyboard', 79.99, 50, 'Electronics')
ON CONFLICT (product_code)
DO UPDATE SET 
    stock = products.stock + EXCLUDED.stock,
    price = EXCLUDED.price,
    updated_at = NOW()
RETURNING *;
```

---

## 성능 최적화 팁

### 대량 데이터 삽입

```sql
-- 좋지 않은 방법 (느림)
BEGIN;
INSERT INTO students (name, email) VALUES ('Student 1', 'student1@email.com');
INSERT INTO students (name, email) VALUES ('Student 2', 'student2@email.com');
-- 1000번 반복...
COMMIT;

-- 좋은 방법 (빠름)
INSERT INTO students (name, email) VALUES
    ('Student 1', 'student1@email.com'),
    ('Student 2', 'student2@email.com'),
    ('Student 3', 'student3@email.com');
    -- 한 번에 여러 개

-- COPY 명령 사용 (가장 빠름)
COPY students (name, email) FROM '/path/to/data.csv' CSV HEADER;
```

### 효율적인 UPDATE

```sql
-- 불필요한 UPDATE 피하기
-- 나쁜 예
UPDATE students SET age = 20;  -- 모든 행 수정

-- 좋은 예
UPDATE students SET age = 20 WHERE age != 20;  -- 필요한 행만 수정
```

### 안전한 DELETE

```sql
-- 항상 WHERE 절 사용
DELETE FROM students WHERE age > 25;

-- 전체 삭제는 TRUNCATE 사용
TRUNCATE TABLE students;  -- DELETE FROM students; 보다 빠름
```

---

## 다음 단계

3단계를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ 데이터 삽입 (INSERT)
✅ 데이터 조회 (SELECT)
✅ 데이터 수정 (UPDATE)
✅ 데이터 삭제 (DELETE)
✅ RETURNING 절 활용
✅ ON CONFLICT (UPSERT)

**다음 학습**: 4단계 - 데이터 조회 심화
- WHERE 절과 조건 연산자
- 정렬 (ORDER BY)과 제한 (LIMIT, OFFSET)
- 집계 함수 (COUNT, SUM, AVG, MAX, MIN)
- GROUP BY와 HAVING
- 서브쿼리
