---
layout: post
title: PostgreSQL 완전 정복 2단계 - 데이터베이스 기본 작업
summary: 데이터베이스 생성/관리, 사용자 권한, 테이블 생성 및 데이터 타입 완벽 가이드
author: keonhee
date: 2025-12-24 09:00:00 +0900
category: Database
keywords: PostgreSQL, Database Management, User Permissions, Data Types, ALTER TABLE
permalink: /blog/postgresql_database_basics/
usemathjax: false
thumbnail: /assets/img/posts/postgresql.png
imageNameKey: postgresql
---



# PostgreSQL 2단계: 데이터베이스 기본 작업

## 목차
1. [데이터베이스 생성/삭제/관리](#1-데이터베이스-생성삭제관리)
2. [사용자 및 권한 관리](#2-사용자-및-권한-관리)
3. [테이블 생성 및 데이터 타입](#3-테이블-생성-및-데이터-타입)
4. [테이블 구조 수정 (ALTER)](#4-테이블-구조-수정-alter)

---

## 1. 데이터베이스 생성/삭제/관리

### 데이터베이스 생성

**기본 생성**

```sql
-- 가장 간단한 형태
CREATE DATABASE mydb;

-- 소유자 지정
CREATE DATABASE mydb OWNER myuser;

-- 인코딩 지정
CREATE DATABASE mydb ENCODING 'UTF8';

-- 템플릿 지정
CREATE DATABASE mydb TEMPLATE template0;
```

**고급 옵션**

```sql
-- 모든 옵션을 포함한 생성
CREATE DATABASE mydb
    OWNER = myuser
    ENCODING = 'UTF8'
    LC_COLLATE = 'ko_KR.UTF-8'
    LC_CTYPE = 'ko_KR.UTF-8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = 100
    TEMPLATE = template0;
```

**옵션 설명**

- `OWNER`: 데이터베이스 소유자
- `ENCODING`: 문자 인코딩 (UTF8 권장)
- `LC_COLLATE`: 문자열 정렬 규칙
- `LC_CTYPE`: 문자 분류 (대소문자 등)
- `TABLESPACE`: 물리적 저장 위치
- `CONNECTION LIMIT`: 최대 동시 연결 수 (-1은 무제한)
- `TEMPLATE`: 복사할 템플릿 데이터베이스

### 데이터베이스 목록 확인

```sql
-- psql 명령어
\l
\l+  -- 상세 정보 포함

-- SQL 쿼리
SELECT datname, datdba, encoding, datcollate, datctype 
FROM pg_database;

-- 크기 포함 조회
SELECT 
    datname AS database_name,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```

### 데이터베이스 수정

```sql
-- 데이터베이스 이름 변경
ALTER DATABASE oldname RENAME TO newname;

-- 소유자 변경
ALTER DATABASE mydb OWNER TO newowner;

-- 연결 제한 변경
ALTER DATABASE mydb CONNECTION LIMIT 50;

-- 기본 설정 변경
ALTER DATABASE mydb SET timezone TO 'Asia/Seoul';
```

### 데이터베이스 연결

```bash
-- psql에서 데이터베이스 변경
\c mydb
\connect mydb

-- 사용자 지정하여 연결
\c mydb myuser

-- 호스트 지정하여 연결
\c "host=localhost dbname=mydb user=myuser"
```

```sql
-- 현재 데이터베이스 확인
SELECT current_database();

-- 현재 사용자 확인
SELECT current_user;

-- 세션 정보 확인
SELECT 
    current_database(),
    current_user,
    inet_client_addr() AS client_ip,
    inet_server_addr() AS server_ip;
```

### 데이터베이스 삭제

```sql
-- 기본 삭제
DROP DATABASE mydb;

-- 존재할 경우에만 삭제
DROP DATABASE IF EXISTS mydb;

-- 활성 연결이 있어도 강제 삭제 (PostgreSQL 13+)
DROP DATABASE mydb WITH (FORCE);
```

**주의사항**

```sql
-- 현재 연결된 데이터베이스는 삭제할 수 없음
-- 다른 데이터베이스로 전환 후 삭제
\c postgres
DROP DATABASE mydb;

-- 다른 사용자가 연결 중인 경우 연결 종료
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'mydb' AND pid <> pg_backend_pid();

-- 그 후 삭제
DROP DATABASE mydb;
```

### 데이터베이스 복사

```sql
-- 템플릿으로 새 데이터베이스 생성
CREATE DATABASE newdb TEMPLATE originaldb;

-- 주의: 원본 데이터베이스에 활성 연결이 없어야 함
```

### 데이터베이스 통계

```sql
-- 데이터베이스 크기
SELECT pg_size_pretty(pg_database_size('mydb'));

-- 모든 데이터베이스 크기
SELECT 
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;

-- 데이터베이스 연결 정보
SELECT 
    datname,
    count(*) AS connections
FROM pg_stat_activity
GROUP BY datname;
```

---

## 2. 사용자 및 권한 관리

### 사용자(역할) 생성

**기본 생성**

```sql
-- 일반 사용자 생성
CREATE USER john WITH PASSWORD 'secure_password';

-- 또는 (USER와 ROLE은 거의 동일)
CREATE ROLE john WITH LOGIN PASSWORD 'secure_password';
```

**권한이 있는 사용자 생성**

```sql
-- 슈퍼유저 생성
CREATE USER admin WITH SUPERUSER PASSWORD 'admin_password';

-- 데이터베이스 생성 권한
CREATE USER developer WITH CREATEDB PASSWORD 'dev_password';

-- 역할 생성 권한
CREATE USER manager WITH CREATEROLE PASSWORD 'mgr_password';

-- 복합 권한
CREATE USER poweruser WITH 
    SUPERUSER 
    CREATEDB 
    CREATEROLE 
    PASSWORD 'power_password';
```

**추가 옵션**

```sql
-- 연결 제한이 있는 사용자
CREATE USER limited_user WITH 
    PASSWORD 'password'
    CONNECTION LIMIT 5;

-- 유효 기간이 있는 사용자
CREATE USER temp_user WITH 
    PASSWORD 'password'
    VALID UNTIL '2025-12-31';

-- 로그인 불가능한 역할 (권한 그룹용)
CREATE ROLE readonly_group;
```

### 사용자 목록 확인

```sql
-- psql 명령어
\du
\du+  -- 상세 정보

-- SQL 쿼리
SELECT 
    rolname AS username,
    rolsuper AS is_superuser,
    rolcreatedb AS can_create_db,
    rolcreaterole AS can_create_role,
    rolcanlogin AS can_login
FROM pg_roles
ORDER BY rolname;
```

### 사용자 수정

```sql
-- 비밀번호 변경
ALTER USER john PASSWORD 'new_password';

-- 슈퍼유저 권한 부여
ALTER USER john SUPERUSER;

-- 슈퍼유저 권한 제거
ALTER USER john NOSUPERUSER;

-- 데이터베이스 생성 권한 부여
ALTER USER john CREATEDB;

-- 연결 제한 변경
ALTER USER john CONNECTION LIMIT 10;

-- 유효 기간 설정
ALTER USER john VALID UNTIL '2025-12-31';

-- 사용자 이름 변경
ALTER USER john RENAME TO john_doe;
```

### 사용자 삭제

```sql
-- 기본 삭제
DROP USER john;

-- 존재할 경우에만 삭제
DROP USER IF EXISTS john;

-- 소유 객체가 있는 경우 먼저 재할당
REASSIGN OWNED BY john TO postgres;
DROP OWNED BY john;
DROP USER john;
```

### 데이터베이스 권한 부여

```sql
-- 데이터베이스 모든 권한 부여
GRANT ALL PRIVILEGES ON DATABASE mydb TO john;

-- 연결 권한만 부여
GRANT CONNECT ON DATABASE mydb TO john;

-- 임시 테이블 생성 권한
GRANT TEMP ON DATABASE mydb TO john;

-- 여러 사용자에게 동시 부여
GRANT ALL PRIVILEGES ON DATABASE mydb TO john, jane, bob;
```

### 테이블 권한 부여

```sql
-- 특정 테이블에 모든 권한
GRANT ALL PRIVILEGES ON TABLE users TO john;

-- SELECT 권한만
GRANT SELECT ON TABLE users TO john;

-- INSERT, UPDATE 권한
GRANT INSERT, UPDATE ON TABLE users TO john;

-- DELETE 권한
GRANT DELETE ON TABLE users TO john;

-- 스키마의 모든 테이블에 권한 부여
GRANT SELECT ON ALL TABLES IN SCHEMA public TO john;

-- 앞으로 생성될 테이블에도 자동 권한 부여
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO john;
```

### 스키마 권한 부여

```sql
-- 스키마 사용 권한
GRANT USAGE ON SCHEMA public TO john;

-- 스키마에서 객체 생성 권한
GRANT CREATE ON SCHEMA public TO john;

-- 모든 권한
GRANT ALL PRIVILEGES ON SCHEMA public TO john;
```

### 시퀀스 권한 부여

```sql
-- 시퀀스 사용 권한 (SERIAL 타입에 필요)
GRANT USAGE, SELECT ON SEQUENCE users_id_seq TO john;

-- 스키마의 모든 시퀀스에 권한 부여
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO john;

-- 기본 권한 설정
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO john;
```

### 권한 확인

```sql
-- 테이블 권한 확인
\dp users
\z users

-- SQL로 확인
SELECT 
    grantee,
    privilege_type
FROM information_schema.table_privileges
WHERE table_name = 'users';

-- 데이터베이스 권한 확인
SELECT 
    datname,
    datacl
FROM pg_database
WHERE datname = 'mydb';
```

### 권한 회수

```sql
-- 테이블 권한 회수
REVOKE SELECT ON TABLE users FROM john;

-- 모든 권한 회수
REVOKE ALL PRIVILEGES ON TABLE users FROM john;

-- 데이터베이스 권한 회수
REVOKE CONNECT ON DATABASE mydb FROM john;

-- 스키마의 모든 테이블 권한 회수
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM john;

-- CASCADE 옵션 (연쇄 회수)
REVOKE ALL PRIVILEGES ON DATABASE mydb FROM john CASCADE;
```

### 역할(Role) 그룹 관리

```sql
-- 읽기 전용 그룹 생성
CREATE ROLE readonly;
GRANT CONNECT ON DATABASE mydb TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT ON TABLES TO readonly;

-- 읽기/쓰기 그룹 생성
CREATE ROLE readwrite;
GRANT CONNECT ON DATABASE mydb TO readwrite;
GRANT USAGE, CREATE ON SCHEMA public TO readwrite;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO readwrite;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO readwrite;

-- 사용자를 그룹에 추가
GRANT readonly TO john;
GRANT readwrite TO jane;

-- 그룹 멤버십 확인
SELECT 
    r.rolname AS role_name,
    m.rolname AS member_name
FROM pg_roles r
JOIN pg_auth_members am ON r.oid = am.roleid
JOIN pg_roles m ON am.member = m.oid
WHERE r.rolname = 'readonly';

-- 그룹에서 제거
REVOKE readonly FROM john;
```

### Row Level Security (행 수준 보안)

```sql
-- RLS 활성화
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- 정책 생성: 사용자는 자신의 주문만 볼 수 있음
CREATE POLICY user_orders_policy ON orders
    FOR SELECT
    USING (user_id = current_user::INTEGER);

-- 정책 생성: 관리자는 모든 데이터를 볼 수 있음
CREATE POLICY admin_all_policy ON orders
    FOR ALL
    TO admin_role
    USING (true);

-- 정책 확인
\d orders

-- RLS 비활성화
ALTER TABLE orders DISABLE ROW LEVEL SECURITY;

-- 정책 삭제
DROP POLICY user_orders_policy ON orders;
```

### 실용적인 권한 관리 예제

**시나리오 1: 애플리케이션 사용자**

```sql
-- 1. 애플리케이션용 사용자 생성
CREATE USER app_user WITH PASSWORD 'app_password';

-- 2. 데이터베이스 연결 권한
GRANT CONNECT ON DATABASE mydb TO app_user;

-- 3. 스키마 사용 권한
GRANT USAGE ON SCHEMA public TO app_user;

-- 4. 테이블 권한
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;

-- 5. 시퀀스 권한
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- 6. 기본 권한 (미래 객체에도 적용)
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO app_user;
```

**시나리오 2: 읽기 전용 분석가**

```sql
-- 1. 분석가 그룹 생성
CREATE ROLE analysts;

-- 2. 읽기 권한만 부여
GRANT CONNECT ON DATABASE analytics_db TO analysts;
GRANT USAGE ON SCHEMA public TO analysts;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analysts;

-- 3. 개별 분석가 생성 및 그룹에 추가
CREATE USER analyst1 WITH PASSWORD 'pass1';
GRANT analysts TO analyst1;

CREATE USER analyst2 WITH PASSWORD 'pass2';
GRANT analysts TO analyst2;
```

---

## 3. 테이블 생성 및 데이터 타입

### 기본 테이블 생성

```sql
-- 간단한 테이블
CREATE TABLE employees (
    id INTEGER,
    name VARCHAR(100),
    hire_date DATE
);

-- 제약조건이 있는 테이블
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE DEFAULT CURRENT_DATE,
    salary DECIMAL(10, 2) CHECK (salary > 0)
);
```

### 숫자형 데이터 타입

```sql
CREATE TABLE numeric_types (
    -- 정수형
    small_num SMALLINT,        -- -32,768 ~ 32,767
    normal_num INTEGER,         -- -2,147,483,648 ~ 2,147,483,647
    big_num BIGINT,            -- -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807
    
    -- 자동 증가 정수
    auto_small SMALLSERIAL,    -- 1 ~ 32,767
    auto_normal SERIAL,        -- 1 ~ 2,147,483,647
    auto_big BIGSERIAL,        -- 1 ~ 9,223,372,036,854,775,807
    
    -- 고정 소수점
    exact_num DECIMAL(10, 2),  -- 정확한 소수점 (10자리, 소수점 2자리)
    numeric_val NUMERIC(15, 4), -- DECIMAL과 동일
    
    -- 부동 소수점
    float_num REAL,            -- 6자리 정밀도
    double_num DOUBLE PRECISION, -- 15자리 정밀도
    
    -- 화폐
    price MONEY                -- 화폐 타입
);
```

**사용 예제**

```sql
INSERT INTO numeric_types (normal_num, exact_num, float_num) VALUES
    (42, 123.45, 3.14159),
    (-100, 999.99, 2.71828);

-- SERIAL 예제
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100),
    price DECIMAL(10, 2)
);
```

### 문자형 데이터 타입

```sql
CREATE TABLE string_types (
    -- 가변 길이 문자열
    short_text VARCHAR(50),     -- 최대 50자
    medium_text VARCHAR(255),   -- 최대 255자
    
    -- 고정 길이 문자열
    fixed_code CHAR(10),        -- 항상 10자 (공백으로 채움)
    
    -- 무제한 길이
    long_text TEXT,             -- 제한 없음
    description TEXT
);
```

**사용 예제**

```sql
INSERT INTO string_types (short_text, fixed_code, long_text) VALUES
    ('Hello', 'ABC123', 'This is a very long text that can contain paragraphs...'),
    ('World', 'XYZ789', 'Another long description...');

-- VARCHAR vs TEXT 비교
CREATE TABLE articles (
    title VARCHAR(200),         -- 제목은 길이 제한
    content TEXT                -- 내용은 무제한
);
```

### 날짜/시간 데이터 타입

```sql
CREATE TABLE datetime_types (
    -- 날짜
    birth_date DATE,            -- 날짜만 (YYYY-MM-DD)
    
    -- 시간
    meeting_time TIME,          -- 시간만 (HH:MM:SS)
    meeting_time_tz TIME WITH TIME ZONE, -- 시간대 포함
    
    -- 날짜와 시간
    created_at TIMESTAMP,       -- 날짜와 시간
    updated_at TIMESTAMP WITH TIME ZONE, -- 시간대 포함 (권장)
    
    -- 시간 간격
    duration INTERVAL           -- 기간
);
```

**사용 예제**

```sql
INSERT INTO datetime_types (birth_date, created_at, duration) VALUES
    ('1990-05-15', '2024-01-15 10:30:00', '2 hours 30 minutes'),
    ('1985-12-25', NOW(), '1 day 3 hours');

-- 현재 시간 함수
SELECT 
    CURRENT_DATE,               -- 오늘 날짜
    CURRENT_TIME,               -- 현재 시간
    CURRENT_TIMESTAMP,          -- 현재 날짜와 시간
    NOW(),                      -- CURRENT_TIMESTAMP와 동일
    LOCALTIMESTAMP;             -- 로컬 타임존

-- 날짜 계산
SELECT 
    CURRENT_DATE + INTERVAL '7 days' AS next_week,
    CURRENT_DATE - INTERVAL '1 month' AS last_month,
    AGE(TIMESTAMP '2000-01-01') AS age_from_2000;
```

### 불리언 타입

```sql
CREATE TABLE settings (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(50),
    is_enabled BOOLEAN DEFAULT false,
    is_public BOOLEAN
);

-- 사용 예제
INSERT INTO settings (feature_name, is_enabled, is_public) VALUES
    ('dark_mode', true, false),
    ('notifications', false, true),
    ('auto_save', TRUE, FALSE);  -- 대소문자 구분 안 함

-- 조회
SELECT * FROM settings WHERE is_enabled = true;
SELECT * FROM settings WHERE is_enabled;  -- true와 동일
SELECT * FROM settings WHERE NOT is_public;
```

### JSON 타입

```sql
CREATE TABLE user_preferences (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50),
    settings JSON,              -- JSON 타입
    metadata JSONB              -- Binary JSON (더 효율적, 권장)
);

-- 사용 예제
INSERT INTO user_preferences (username, settings, metadata) VALUES
    ('john', 
     '{"theme": "dark", "language": "ko"}',
     '{"last_login": "2024-01-15", "notifications": true}'),
    ('jane',
     '{"theme": "light", "language": "en", "fontSize": 14}',
     '{"last_login": "2024-01-14", "notifications": false}');

-- JSON 데이터 조회
SELECT 
    username,
    settings->>'theme' AS theme,
    metadata->>'last_login' AS last_login
FROM user_preferences;

-- JSONB 조건 검색
SELECT * FROM user_preferences 
WHERE metadata @> '{"notifications": true}';
```

### 배열 타입

```sql
CREATE TABLE blog_posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    tags TEXT[],                -- 문자열 배열
    ratings INTEGER[]           -- 정수 배열
);

-- 사용 예제
INSERT INTO blog_posts (title, tags, ratings) VALUES
    ('PostgreSQL Guide', 
     ARRAY['database', 'postgresql', 'tutorial'],
     ARRAY[5, 4, 5, 5]),
    ('Python Tips',
     '{"python", "programming", "tips"}',  -- 다른 표기법
     '{4, 5, 3, 4}');

-- 배열 조회
SELECT 
    title,
    tags[1] AS first_tag,       -- 배열 인덱스는 1부터 시작
    array_length(tags, 1) AS tag_count
FROM blog_posts;

-- 배열 검색
SELECT * FROM blog_posts WHERE 'postgresql' = ANY(tags);
SELECT * FROM blog_posts WHERE tags @> ARRAY['python'];
```

### UUID 타입

```sql
-- UUID 확장 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 사용 예제
INSERT INTO sessions (user_id) VALUES (1), (2);

SELECT * FROM sessions;
-- session_id는 자동으로 UUID 생성됨
-- 예: 550e8400-e29b-41d4-a716-446655440000
```

### 기타 특수 타입

```sql
CREATE TABLE special_types (
    -- 네트워크 주소
    ip_address INET,            -- IP 주소
    mac_address MACADDR,        -- MAC 주소
    
    -- 기하학
    location POINT,             -- 좌표 (x, y)
    area BOX,                   -- 직사각형
    path_data PATH,             -- 경로
    
    -- 비트 문자열
    flags BIT(8),              -- 고정 길이 비트
    permissions BIT VARYING(16), -- 가변 길이 비트
    
    -- 범위 타입
    valid_period TSRANGE,       -- 타임스탬프 범위
    price_range INT4RANGE,      -- 정수 범위
    
    -- XML
    document XML,
    
    -- 바이너리 데이터
    file_data BYTEA
);

-- 사용 예제
INSERT INTO special_types (ip_address, location, flags) VALUES
    ('192.168.1.1', '(10.5, 20.3)', B'10101010'),
    ('10.0.0.1', POINT(30.2, 40.7), B'11110000');
```

### 복합 타입 (사용자 정의)

```sql
-- 복합 타입 생성
CREATE TYPE address AS (
    street VARCHAR(100),
    city VARCHAR(50),
    zipcode VARCHAR(10)
);

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    home_address address,
    work_address address
);

-- 사용 예제
INSERT INTO customers (name, home_address, work_address) VALUES
    ('John Doe',
     ROW('123 Main St', 'Seoul', '12345'),
     ROW('456 Office Rd', 'Seoul', '67890'));

-- 조회
SELECT 
    name,
    (home_address).city AS home_city,
    (work_address).street AS work_street
FROM customers;
```

### ENUM 타입

```sql
-- ENUM 타입 생성
CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral');
CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'delivered', 'cancelled');

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    status order_status DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 사용 예제
INSERT INTO orders (status) VALUES
    ('pending'),
    ('processing'),
    ('shipped');

-- 조회
SELECT * FROM orders WHERE status = 'pending';

-- ENUM 값 확인
SELECT enum_range(NULL::order_status);
```

---

## 4. 테이블 구조 수정 (ALTER)

### 열 추가

```sql
-- 기본 열 추가
ALTER TABLE employees ADD COLUMN department VARCHAR(50);

-- NOT NULL 제약과 기본값 포함
ALTER TABLE employees 
ADD COLUMN salary DECIMAL(10, 2) NOT NULL DEFAULT 0;

-- 여러 열 동시 추가
ALTER TABLE employees 
ADD COLUMN phone VARCHAR(20),
ADD COLUMN address TEXT;
```

### 열 삭제

```sql
-- 열 삭제
ALTER TABLE employees DROP COLUMN phone;

-- 존재할 경우에만 삭제
ALTER TABLE employees DROP COLUMN IF EXISTS phone;

-- CASCADE: 종속 객체도 함께 삭제
ALTER TABLE employees DROP COLUMN address CASCADE;

-- RESTRICT: 종속 객체가 있으면 실패 (기본값)
ALTER TABLE employees DROP COLUMN address RESTRICT;
```

### 열 이름 변경

```sql
-- 열 이름 변경
ALTER TABLE employees RENAME COLUMN dept TO department;

-- 여러 열 이름 변경
ALTER TABLE employees RENAME COLUMN emp_name TO name;
ALTER TABLE employees RENAME COLUMN emp_email TO email;
```

### 데이터 타입 변경

```sql
-- 타입 변경
ALTER TABLE employees ALTER COLUMN salary TYPE NUMERIC(12, 2);

-- USING 절로 변환 로직 지정
ALTER TABLE employees 
ALTER COLUMN hire_date TYPE TIMESTAMP 
USING hire_date::TIMESTAMP;

-- VARCHAR 길이 변경
ALTER TABLE employees ALTER COLUMN name TYPE VARCHAR(200);

-- 예제: 문자열을 정수로 변환
ALTER TABLE products 
ALTER COLUMN code TYPE INTEGER 
USING code::INTEGER;
```

### NOT NULL 제약 조건

```sql
-- NOT NULL 추가
ALTER TABLE employees ALTER COLUMN email SET NOT NULL;

-- NOT NULL 제거
ALTER TABLE employees ALTER COLUMN email DROP NOT NULL;

-- NULL 값이 있는 경우 먼저 처리
UPDATE employees SET email = 'unknown@company.com' WHERE email IS NULL;
ALTER TABLE employees ALTER COLUMN email SET NOT NULL;
```

### 기본값 설정

```sql
-- 기본값 설정
ALTER TABLE employees ALTER COLUMN created_at SET DEFAULT NOW();

-- 기본값 제거
ALTER TABLE employees ALTER COLUMN created_at DROP DEFAULT;

-- 기존 행에는 영향 없음, 새로 삽입되는 행에만 적용
ALTER TABLE employees ALTER COLUMN status SET DEFAULT 'active';
```

### 제약조건 추가

```sql
-- PRIMARY KEY 추가
ALTER TABLE employees ADD PRIMARY KEY (id);

-- UNIQUE 제약 추가
ALTER TABLE employees ADD UNIQUE (email);

-- 이름 지정하여 UNIQUE 제약 추가
ALTER TABLE employees 
ADD CONSTRAINT unique_email UNIQUE (email);

-- CHECK 제약 추가
ALTER TABLE employees 
ADD CONSTRAINT check_salary CHECK (salary > 0);

-- 복합 제약조건
ALTER TABLE employees 
ADD CONSTRAINT check_dates CHECK (end_date > start_date);

-- FOREIGN KEY 추가
ALTER TABLE orders 
ADD CONSTRAINT fk_customer 
FOREIGN KEY (customer_id) REFERENCES customers(id);

-- CASCADE 옵션과 함께
ALTER TABLE orders 
ADD CONSTRAINT fk_customer 
FOREIGN KEY (customer_id) REFERENCES customers(id)
ON DELETE CASCADE 
ON UPDATE CASCADE;
```

### 제약조건 삭제

```sql
-- 제약조건 삭제 (이름으로)
ALTER TABLE employees DROP CONSTRAINT unique_email;

-- 존재할 경우에만 삭제
ALTER TABLE employees DROP CONSTRAINT IF EXISTS unique_email;

-- PRIMARY KEY 삭제
ALTER TABLE employees DROP CONSTRAINT employees_pkey;
```

### 테이블 이름 변경

```sql
-- 테이블 이름 변경
ALTER TABLE employees RENAME TO staff;

-- 스키마 간 이동
ALTER TABLE staff SET SCHEMA hr;
```

### 소유자 변경

```sql
-- 테이블 소유자 변경
ALTER TABLE employees OWNER TO new_owner;
```

### 실전 예제

**예제 1: 기존 테이블에 타임스탬프 추가**

```sql
-- 1. created_at 열 추가
ALTER TABLE products 
ADD COLUMN created_at TIMESTAMP DEFAULT NOW();

-- 2. updated_at 열 추가
ALTER TABLE products 
ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();

-- 3. 기존 행의 created_at 값 설정
UPDATE products SET created_at = NOW() WHERE created_at IS NULL;

-- 4. NOT NULL 제약 추가
ALTER TABLE products ALTER COLUMN created_at SET NOT NULL;
ALTER TABLE products ALTER COLUMN updated_at SET NOT NULL;
```

**예제 2: 외래 키 관계 추가**

```sql
-- 1. 먼저 참조 테이블 생성
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(100)
);

-- 2. employees 테이블에 department_id 열 추가
ALTER TABLE employees ADD COLUMN department_id INTEGER;

-- 3. 외래 키 제약 추가
ALTER TABLE employees 
ADD CONSTRAINT fk_department 
FOREIGN KEY (department_id) REFERENCES departments(dept_id)
ON DELETE SET NULL;
```

**예제 3: 테이블 리팩토링**

```sql
-- 기존 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    fullname VARCHAR(200),
    contact VARCHAR(100)
);

-- 1. 이름을 first_name과 last_name으로 분리
ALTER TABLE users ADD COLUMN first_name VARCHAR(100);
ALTER TABLE users ADD COLUMN last_name VARCHAR(100);

-- 2. 기존 데이터 분리 (간단한 예제)
UPDATE users 
SET first_name = split_part(fullname, ' ', 1),
    last_name = split_part(fullname, ' ', 2);

-- 3. 기존 열 삭제
ALTER TABLE users DROP COLUMN fullname;

-- 4. contact를 email과 phone으로 분리
ALTER TABLE users ADD COLUMN email VARCHAR(100);
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 데이터 마이그레이션 후
ALTER TABLE users DROP COLUMN contact;
```

**예제 4: 제약조건 추가하기**

```sql
-- 기존 products 테이블
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10, 2),
    stock INTEGER,
    category VARCHAR(50)
);

-- 1. price는 0보다 커야 함
ALTER TABLE products 
ADD CONSTRAINT check_price_positive CHECK (price > 0);

-- 2. stock은 0 이상이어야 함
ALTER TABLE products 
ADD CONSTRAINT check_stock_non_negative CHECK (stock >= 0);

-- 3. name은 고유해야 함
ALTER TABLE products ADD UNIQUE (name);

-- 4. category는 NULL이 될 수 없음
UPDATE products SET category = 'General' WHERE category IS NULL;
ALTER TABLE products ALTER COLUMN category SET NOT NULL;
```

---

## 실습 과제

### 과제 1: 회사 데이터베이스 만들기

```sql
-- 1. 데이터베이스 생성
CREATE DATABASE company_db;
\c company_db

-- 2. 부서 테이블 생성
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(100) NOT NULL UNIQUE,
    location VARCHAR(100)
);

-- 3. 직원 테이블 생성
CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    hire_date DATE NOT NULL DEFAULT CURRENT_DATE,
    salary DECIMAL(10, 2) CHECK (salary > 0),
    department_id INTEGER REFERENCES departments(dept_id)
);

-- 4. 테이블 확인
\dt
\d employees
\d departments
```

### 과제 2: 테이블 수정 연습

```sql
-- 1. employees 테이블에 manager_id 열 추가
ALTER TABLE employees 
ADD COLUMN manager_id INTEGER REFERENCES employees(emp_id);

-- 2. status 열 추가 (기본값: 'active')
ALTER TABLE employees 
ADD COLUMN status VARCHAR(20) DEFAULT 'active';

-- 3. updated_at 열 추가
ALTER TABLE employees 
ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();

-- 4. email은 필수값으로 변경
ALTER TABLE employees ALTER COLUMN email SET NOT NULL;

-- 5. 테이블 구조 확인
\d+ employees
```

### 과제 3: 사용자 및 권한 설정

```sql
-- 1. 개발자 사용자 생성
CREATE USER developer WITH PASSWORD 'dev_pass';

-- 2. 읽기 전용 사용자 생성
CREATE USER analyst WITH PASSWORD 'analyst_pass';

-- 3. 개발자에게 모든 권한 부여
GRANT ALL PRIVILEGES ON DATABASE company_db TO developer;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO developer;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO developer;

-- 4. 분석가에게 읽기 권한만 부여
GRANT CONNECT ON DATABASE company_db TO analyst;
GRANT USAGE ON SCHEMA public TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;

-- 5. 권한 확인
\du
```

---

## 다음 단계

2단계를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ 데이터베이스 생성, 수정, 삭제
✅ 사용자 관리 및 권한 부여
✅ 다양한 데이터 타입 사용
✅ 테이블 구조 수정

**다음 학습**: 3단계 - 데이터 조작 (CRUD)
- INSERT: 데이터 삽입
- SELECT: 데이터 조회 기초
- UPDATE: 데이터 수정
- DELETE: 데이터 삭제
