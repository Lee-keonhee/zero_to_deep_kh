---
layout: post
title: PostgreSQL 완전 정복 1단계 - 기초부터 시작하기
summary: PostgreSQL 설치부터 기본 개념, psql 명령어까지 완벽 가이드
author: keonhee
date: 2025-12-24 09:00:00 +0900
category: Database
keywords: PostgreSQL, Database, SQL, psql, 설치, 기초
permalink: /blog/postgresql_basics/
usemathjax: false
thumbnail: /assets/img/posts/postgresql_basics.png
---
# PostgreSQL 1단계: 기초

## 목차
1. [PostgreSQL 소개 및 특징](#1-postgresql-소개-및-특징)
2. [설치 및 환경 설정](#2-설치-및-환경-설정)
3. [기본 개념](#3-기본-개념)
4. [psql 명령어 익히기](#4-psql-명령어-익히기)

---

## 1. PostgreSQL 소개 및 특징

### PostgreSQL이란?

PostgreSQL은 세계에서 가장 진보된 오픈소스 관계형 데이터베이스 관리 시스템(RDBMS)입니다. 1986년 캘리포니아 대학교 버클리에서 시작된 POSTGRES 프로젝트에서 발전했습니다.

### 주요 특징

**1. 오픈소스**
- 완전 무료
- PostgreSQL 라이선스 (BSD 라이선스와 유사)
- 상업적 사용 가능

**2. ACID 준수**
- **A**tomicity (원자성): 트랜잭션의 모든 작업이 완료되거나 모두 실패
- **C**onsistency (일관성): 데이터베이스가 항상 일관된 상태 유지
- **I**solation (격리성): 동시 트랜잭션이 서로 영향을 주지 않음
- **D**urability (지속성): 커밋된 데이터는 영구적으로 보존

**3. 확장성**
- 사용자 정의 데이터 타입
- 사용자 정의 함수
- 외부 데이터 래퍼 (Foreign Data Wrapper)
- 다양한 확장 모듈 (Extension)

**4. 고급 기능**
- 복잡한 쿼리 지원
- 외래 키 (Foreign Key)
- 트리거 (Trigger)
- 뷰 (View)
- 트랜잭션 무결성
- 멀티 버전 동시성 제어 (MVCC)

**5. 다양한 데이터 타입**
- 기본 타입 (정수, 실수, 문자열, 날짜/시간)
- JSON/JSONB
- XML
- 배열
- UUID
- 기하학적 데이터
- 네트워크 주소

### PostgreSQL vs 다른 데이터베이스

| 기능 | PostgreSQL | MySQL | Oracle | SQL Server |
|------|-----------|-------|--------|-----------|
| 오픈소스 | ✅ | ✅ | ❌ | ❌ |
| ACID 완전 준수 | ✅ | 부분적 | ✅ | ✅ |
| JSON 지원 | ✅ | ✅ | ✅ | ✅ |
| 윈도우 함수 | ✅ | ✅ | ✅ | ✅ |
| CTE (WITH) | ✅ | ✅ | ✅ | ✅ |
| 라이선스 비용 | 무료 | 무료 | 유료 | 유료 |

### 사용 사례

1. **웹 애플리케이션**
   - Django, Rails, Node.js 등과 통합
   - RESTful API 백엔드

2. **지리정보 시스템 (GIS)**
   - PostGIS 확장을 통한 공간 데이터 처리
   - 지도 애플리케이션

3. **데이터 웨어하우스**
   - 대용량 데이터 분석
   - 비즈니스 인텔리전스

4. **금융 시스템**
   - 높은 트랜잭션 무결성 요구
   - 정확한 데이터 관리

---

## 2. 설치 및 환경 설정

### Windows 설치

**방법 1: 공식 설치 프로그램**

1. 공식 웹사이트 방문
   ```
   https://www.postgresql.org/download/windows/
   ```

2. EDB 설치 프로그램 다운로드
   - 최신 버전 선택 (예: PostgreSQL 16)
   - Windows x86-64 선택

3. 설치 진행
   ```
   단계 1: 설치 디렉토리 선택
   예: C:\Program Files\PostgreSQL\16
   
   단계 2: 구성요소 선택
   ✅ PostgreSQL Server
   ✅ pgAdmin 4 (GUI 관리 도구)
   ✅ Stack Builder (추가 도구)
   ✅ Command Line Tools
   
   단계 3: 데이터 디렉토리 선택
   예: C:\Program Files\PostgreSQL\16\data
   
   단계 4: 슈퍼유저 비밀번호 설정
   postgres 사용자의 비밀번호 입력
   
   단계 5: 포트 번호 설정
   기본값: 5432
   
   단계 6: 로케일 선택
   기본값: [Default locale]
   
   단계 7: 설치 시작
   ```

4. 설치 확인
   ```cmd
   # 명령 프롬프트에서
   psql --version
   ```

**방법 2: Scoop 사용 (Windows 패키지 매니저)**

```powershell
# Scoop 설치 (미설치 시)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# PostgreSQL 설치
scoop install postgresql
```

### macOS 설치

**방법 1: Homebrew**

```bash
# Homebrew 설치 확인
brew --version

# PostgreSQL 설치
brew install postgresql@16

# 설치 확인
postgres --version

# 서비스 시작
brew services start postgresql@16

# 서비스 중지
brew services stop postgresql@16

# 서비스 재시작
brew services restart postgresql@16
```

**방법 2: Postgres.app**

1. https://postgresapp.com/ 방문
2. Postgres.app 다운로드 및 설치
3. Applications 폴더로 이동
4. Postgres.app 실행

### Linux 설치

**Ubuntu/Debian**

```bash
# 패키지 목록 업데이트
sudo apt update

# PostgreSQL 설치
sudo apt install postgresql postgresql-contrib

# 서비스 상태 확인
sudo systemctl status postgresql

# 서비스 시작
sudo systemctl start postgresql

# 서비스 활성화 (부팅 시 자동 시작)
sudo systemctl enable postgresql

# 버전 확인
psql --version
```

**CentOS/RHEL/Fedora**

```bash
# PostgreSQL 저장소 추가
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-8-x86_64/pgdg-redhat-repo-latest.noarch.rpm

# PostgreSQL 16 설치
sudo dnf install -y postgresql16-server postgresql16

# 데이터베이스 초기화
sudo /usr/pgsql-16/bin/postgresql-16-setup initdb

# 서비스 시작
sudo systemctl start postgresql-16

# 서비스 활성화
sudo systemctl enable postgresql-16
```

### Docker를 이용한 설치

```bash
# PostgreSQL 이미지 다운로드
docker pull postgres:16

# PostgreSQL 컨테이너 실행
docker run --name my-postgres \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  -v pgdata:/var/lib/postgresql/data \
  -d postgres:16

# 컨테이너 상태 확인
docker ps

# PostgreSQL 접속
docker exec -it my-postgres psql -U myuser -d mydb

# 컨테이너 중지
docker stop my-postgres

# 컨테이너 시작
docker start my-postgres

# 컨테이너 삭제
docker rm my-postgres
```

**docker-compose.yml 예제**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16
    container_name: my-postgres
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mysecretpassword
      POSTGRES_DB: mydb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  pgdata:
```

실행:
```bash
docker-compose up -d
```

### 초기 설정

**1. postgres 사용자로 접속 (Linux/macOS)**

```bash
# postgres 사용자로 전환
sudo -u postgres psql

# 또는 바로 psql 실행
sudo -u postgres psql
```

**2. 비밀번호 변경**

```sql
-- postgres 사용자 비밀번호 변경
ALTER USER postgres PASSWORD 'newpassword';
```

**3. 새로운 사용자 생성**

```sql
-- 슈퍼유저 생성
CREATE USER myuser WITH SUPERUSER PASSWORD 'mypassword';

-- 일반 사용자 생성
CREATE USER normaluser WITH PASSWORD 'password123';
```

**4. 데이터베이스 생성**

```sql
-- 데이터베이스 생성
CREATE DATABASE mydb OWNER myuser;
```

**5. 외부 접속 허용 설정**

`postgresql.conf` 파일 수정:
```conf
# 모든 IP에서 접속 허용
listen_addresses = '*'
```

`pg_hba.conf` 파일 수정:
```conf
# IPv4 로컬 연결
host    all             all             0.0.0.0/0               md5

# IPv6 로컬 연결
host    all             all             ::/0                    md5
```

설정 파일 위치 찾기:
```sql
SHOW config_file;
SHOW hba_file;
```

---

## 3. 기본 개념

### 데이터베이스 구조 계층

```
PostgreSQL 서버 (Instance)
│
├── 데이터베이스 1 (Database)
│   ├── 스키마 (Schema) - public (기본)
│   │   ├── 테이블 (Table)
│   │   ├── 뷰 (View)
│   │   ├── 인덱스 (Index)
│   │   ├── 시퀀스 (Sequence)
│   │   ├── 함수 (Function)
│   │   └── 프로시저 (Procedure)
│   │
│   ├── 스키마 2
│   └── 스키마 3
│
├── 데이터베이스 2
└── 데이터베이스 3
```

### 주요 구성 요소

**1. 데이터베이스 (Database)**
- 독립적인 데이터 저장 공간
- 여러 스키마를 포함
- 서로 격리되어 있음

**2. 스키마 (Schema)**
- 데이터베이스 내의 네임스페이스
- 테이블, 뷰 등을 논리적으로 그룹화
- 기본 스키마: `public`

**3. 테이블 (Table)**
- 실제 데이터를 저장하는 구조
- 행(Row)과 열(Column)로 구성
- 관계형 데이터베이스의 핵심

**4. 행 (Row/Tuple)**
- 테이블의 개별 레코드
- 하나의 데이터 항목

**5. 열 (Column/Attribute)**
- 테이블의 필드
- 데이터 타입을 가짐

**6. 기본 키 (Primary Key)**
- 행을 고유하게 식별하는 열
- 중복 불가, NULL 불가

**7. 외래 키 (Foreign Key)**
- 다른 테이블의 기본 키를 참조
- 테이블 간의 관계 정의

### 테이블 예제

```
users 테이블
┌────┬──────────┬─────────────────────┬────────────────────┐
│ id │ username │ email               │ created_at         │
├────┼──────────┼─────────────────────┼────────────────────┤
│ 1  │ john     │ john@example.com    │ 2024-01-15 10:30   │
│ 2  │ jane     │ jane@example.com    │ 2024-01-16 14:20   │
│ 3  │ bob      │ bob@example.com     │ 2024-01-17 09:15   │
└────┴──────────┴─────────────────────┴────────────────────┘
  ↑       ↑            ↑                      ↑
  │       │            │                      └─ 열 (Column)
  │       │            └─ 열 (Column)
  │       └─ 열 (Column)
  └─ 기본 키 (Primary Key)

각 행(Row)은 하나의 사용자 정보
```

### 스키마 개념

```sql
-- 스키마 생성
CREATE SCHEMA sales;
CREATE SCHEMA hr;

-- 스키마에 테이블 생성
CREATE TABLE sales.orders (
    order_id SERIAL PRIMARY KEY,
    amount DECIMAL(10, 2)
);

CREATE TABLE hr.employees (
    emp_id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 스키마 지정하여 조회
SELECT * FROM sales.orders;
SELECT * FROM hr.employees;

-- 기본 스키마 (public)
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY
);
-- 이것은 public.products와 동일
```

### 데이터 무결성

**1. 개체 무결성 (Entity Integrity)**
- 기본 키는 NULL이 될 수 없음
- 기본 키는 중복될 수 없음

**2. 참조 무결성 (Referential Integrity)**
- 외래 키는 참조하는 테이블의 기본 키 값만 가능
- 존재하지 않는 값을 참조할 수 없음

**3. 도메인 무결성 (Domain Integrity)**
- 열은 정의된 데이터 타입만 저장 가능
- CHECK 제약조건으로 값의 범위 제한

---

## 4. psql 명령어 익히기

### psql 접속

```bash
# 기본 접속 (로컬 호스트, postgres 사용자)
psql -U postgres

# 특정 데이터베이스로 접속
psql -U postgres -d mydb

# 호스트와 포트 지정
psql -h localhost -p 5432 -U myuser -d mydb

# 비밀번호 입력 프롬프트 표시
psql -U postgres -W

# 연결 문자열 사용
psql "postgresql://myuser:mypassword@localhost:5432/mydb"
```

### 기본 메타 명령어

**데이터베이스 관련**

```sql
-- 데이터베이스 목록 보기
\l
\list

-- 현재 데이터베이스 확인
SELECT current_database();

-- 데이터베이스 연결 변경
\c mydb
\connect mydb

-- 데이터베이스 크기 확인
\l+
```

**테이블 관련**

```sql
-- 현재 스키마의 테이블 목록
\dt

-- 모든 스키마의 테이블 목록
\dt *.*

-- 테이블 구조 확인
\d tablename
\d users

-- 상세 정보 포함
\d+ users

-- 특정 스키마의 테이블
\dt sales.*
```

**스키마 관련**

```sql
-- 스키마 목록
\dn

-- 현재 스키마 확인
SHOW search_path;

-- 스키마 변경
SET search_path TO sales, public;
```

**뷰 관련**

```sql
-- 뷰 목록
\dv

-- 뷰 상세 정보
\d+ viewname
```

**인덱스 관련**

```sql
-- 인덱스 목록
\di

-- 특정 테이블의 인덱스
\d users
```

**함수 관련**

```sql
-- 함수 목록
\df

-- 함수 상세 정보
\df+ functionname
```

**사용자/역할 관련**

```sql
-- 사용자(역할) 목록
\du

-- 사용자 상세 정보
\du+

-- 현재 사용자 확인
SELECT current_user;
```

### SQL 실행 명령어

```sql
-- SQL 파일 실행
\i /path/to/file.sql
\include /path/to/file.sql

-- 쿼리 결과를 파일로 저장
\o /path/to/output.txt
SELECT * FROM users;
\o  -- 파일 출력 종료

-- CSV로 내보내기
\copy users TO '/path/to/users.csv' CSV HEADER;

-- CSV에서 가져오기
\copy users FROM '/path/to/users.csv' CSV HEADER;
```

### 출력 형식 설정

```sql
-- 확장된 디스플레이 모드 (세로 형식)
\x
\x auto  -- 자동
\x off   -- 해제

-- 예제
\x
SELECT * FROM users WHERE id = 1;

-- 출력 예:
-[ RECORD 1 ]------------------
id         | 1
username   | john
email      | john@example.com
created_at | 2024-01-15 10:30

-- 테이블 형식으로 복귀
\x off

-- 페이징 설정
\pset pager off  -- 페이저 비활성화
\pset pager on   -- 페이저 활성화

-- NULL 값 표시 설정
\pset null '[NULL]'

-- 테두리 스타일
\pset border 2

-- 출력 형식
\pset format aligned    -- 정렬된 형식 (기본)
\pset format unaligned  -- 정렬되지 않은 형식
\pset format wrapped    -- 줄바꿈 형식
\pset format html       -- HTML 형식
```

### 유용한 명령어

```sql
-- 이전 명령어 편집
\e
\edit

-- 마지막 쿼리 재실행
\g

-- 쿼리 실행 시간 표시
\timing
\timing on
\timing off

-- 명령어 히스토리 확인
\s

-- 히스토리를 파일로 저장
\s /path/to/history.txt

-- 도움말
\?          -- psql 명령어 도움말
\h          -- SQL 명령어 도움말
\h SELECT   -- 특정 SQL 명령어 도움말

-- 인코딩 확인
\encoding

-- 버전 확인
SELECT version();

-- psql 종료
\q
\quit
exit
```

### 쿼리 결과 포맷팅 예제

```sql
-- 기본 쿼리
SELECT id, username, email FROM users;

 id | username |      email       
----+----------+------------------
  1 | john     | john@example.com
  2 | jane     | jane@example.com
  3 | bob      | bob@example.com

-- 확장 모드
\x
SELECT * FROM users WHERE id = 1;

-[ RECORD 1 ]------------------
id         | 1
username   | john
email      | john@example.com
created_at | 2024-01-15 10:30:00

-- CSV 형식
\pset format unaligned
\pset fieldsep ','
SELECT id, username, email FROM users;

id,username,email
1,john,john@example.com
2,jane,jane@example.com
3,bob,bob@example.com
```

### 환경 설정 파일 (.psqlrc)

홈 디렉토리에 `.psqlrc` 파일 생성:

```bash
# ~/.psqlrc

-- 타이밍 자동 활성화
\timing

-- NULL 값 표시
\pset null '[NULL]'

-- 히스토리 크기
\set HISTSIZE 10000

-- 에러 발생 시 중단
\set ON_ERROR_STOP on

-- 프롬프트 커스터마이징
\set PROMPT1 '%n@%/%R%# '

-- 자동 완성 활성화
\set COMP_KEYWORD_CASE upper
```

### 변수 사용

```sql
-- 변수 설정
\set myvar 100

-- 변수 사용
SELECT * FROM users WHERE id = :myvar;

-- 현재 날짜 변수
\set today '2024-01-15'
SELECT * FROM orders WHERE order_date = :'today';

-- 변수 목록 확인
\set
```

### 트랜잭션 제어

```sql
-- 자동 커밋 비활성화
\set AUTOCOMMIT off

-- 이후 모든 명령은 트랜잭션 내에서 실행
INSERT INTO users (username, email) VALUES ('test', 'test@example.com');

-- 커밋
COMMIT;

-- 또는 롤백
ROLLBACK;

-- 자동 커밋 재활성화
\set AUTOCOMMIT on
```

---

## 실습 예제

### 1. PostgreSQL 접속 연습

```bash
# 1. postgres 사용자로 접속
psql -U postgres

# 2. 데이터베이스 목록 확인
\l

# 3. 새 데이터베이스 생성
CREATE DATABASE practice_db;

# 4. 생성한 데이터베이스로 연결
\c practice_db

# 5. 현재 데이터베이스 확인
SELECT current_database();
```

### 2. 테이블 생성 및 확인

```sql
-- 1. 테이블 생성
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INTEGER
);

-- 2. 테이블 목록 확인
\dt

-- 3. 테이블 구조 확인
\d students

-- 4. 상세 정보 확인
\d+ students
```

### 3. psql 명령어 연습

```sql
-- 1. 타이밍 활성화
\timing

-- 2. 확장 모드 켜기
\x

-- 3. NULL 표시 설정
\pset null '<NULL>'

-- 4. 설정 확인
\pset

-- 5. 도움말 확인
\?
\h CREATE TABLE
```

---

## 다음 단계

1단계를 완료했습니다! 이제 다음을 할 수 있습니다:

✅ PostgreSQL 설치 및 실행
✅ psql로 데이터베이스 접속
✅ 기본 메타 명령어 사용
✅ 데이터베이스 구조 이해

**다음 학습**: 2단계 - 데이터베이스 기본 작업
- 데이터베이스 생성/삭제/관리
- 사용자 및 권한 관리
- 테이블 생성 및 데이터 타입
- 테이블 구조 수정
