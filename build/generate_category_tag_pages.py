#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카테고리와 태그 페이지를 자동으로 생성하는 스크립트
사용법: python generate_category_tag_pages.py
"""

import os
import re
from pathlib import Path

def extract_frontmatter(content):
    """포스트의 Front Matter에서 category와 tags 추출"""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return None, []
    
    frontmatter = match.group(1)
    
    # category 추출
    category_match = re.search(r'^category:\s*(.+)$', frontmatter, re.MULTILINE)
    category = category_match.group(1).strip() if category_match else None
    
    # tags 추출
    tags_match = re.search(r'^tags:\s*\[(.*?)\]', frontmatter, re.MULTILINE)
    tags = []
    if tags_match:
        tags_str = tags_match.group(1)
        tags = [tag.strip().strip('"').strip("'") for tag in tags_str.split(',')]
    
    return category, tags

def get_all_categories_and_tags():
    """모든 포스트를 읽어서 카테고리와 태그 목록 생성"""
    posts_dir = Path('_posts')
    if not posts_dir.exists():
        print("❌ _posts 폴더를 찾을 수 없습니다!")
        return set(), set()
    
    categories = set()
    tags = set()
    
    for post_file in posts_dir.glob('*.md'):
        try:
            content = post_file.read_text(encoding='utf-8')
            category, post_tags = extract_frontmatter(content)
            
            if category:
                categories.add(category)
            if post_tags:
                tags.update(post_tags)
                
        except Exception as e:
            print(f"⚠️  {post_file.name} 읽기 실패: {e}")
    
    return categories, tags

def create_category_page(category):
    """카테고리 페이지 생성"""
    content = f"""---
layout: category
title: {category}
category: {category}
permalink: /blog/categories/{category}/
---
"""
    
    category_dir = Path('category')
    category_dir.mkdir(exist_ok=True)
    
    # 파일명에 사용할 수 없는 문자 제거
    safe_filename = re.sub(r'[<>:"/\\|?*]', '-', category)
    file_path = category_dir / f'{safe_filename}.md'
    
    file_path.write_text(content, encoding='utf-8')
    return file_path

def create_tag_page(tag):
    """태그 페이지 생성"""
    content = f"""---
layout: tag
title: {tag}
tag: {tag}
permalink: /blog/tags/{tag}/
---
"""
    
    tag_dir = Path('tag')
    tag_dir.mkdir(exist_ok=True)
    
    # 파일명에 사용할 수 없는 문자 제거
    safe_filename = re.sub(r'[<>:"/\\|?*]', '-', tag)
    file_path = tag_dir / f'{safe_filename}.md'
    
    file_path.write_text(content, encoding='utf-8')
    return file_path

def main():
    print("🔍 포스트에서 카테고리와 태그 수집 중...")
    categories, tags = get_all_categories_and_tags()
    
    if not categories and not tags:
        print("❌ 카테고리나 태그를 찾을 수 없습니다!")
        return
    
    print(f"\n📁 발견된 카테고리: {len(categories)}개")
    for cat in sorted(categories):
        print(f"  - {cat}")
    
    print(f"\n🏷️  발견된 태그: {len(tags)}개")
    for tag in sorted(tags):
        print(f"  - {tag}")
    
    print("\n📝 카테고리 페이지 생성 중...")
    for category in categories:
        try:
            file_path = create_category_page(category)
            print(f"  ✅ {file_path}")
        except Exception as e:
            print(f"  ❌ {category} 실패: {e}")
    
    print("\n📝 태그 페이지 생성 중...")
    for tag in tags:
        try:
            file_path = create_tag_page(tag)
            print(f"  ✅ {file_path}")
        except Exception as e:
            print(f"  ❌ {tag} 실패: {e}")
    
    print("\n✨ 완료! Jekyll을 재시작하세요.")
    print("   bundle exec jekyll serve")

if __name__ == '__main__':
    main()
