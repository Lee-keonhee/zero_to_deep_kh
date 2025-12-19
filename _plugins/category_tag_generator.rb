# Category and Tag Page Generator
# 
# 이 플러그인은 _posts의 category와 tags를 자동으로 읽어서
# category/[카테고리명].md 와 tag/[태그명].md 파일을 생성합니다.

module Jekyll
  class CategoryPageGenerator < Generator
    safe true

    def generate(site)
      if site.layouts.key? 'category'
        dir = 'category'
        site.categories.each_key do |category|
          site.pages << CategoryPage.new(site, site.source, File.join(dir, category), category)
        end
      end
    end
  end

  class CategoryPage < Page
    def initialize(site, base, dir, category)
      @site = site
      @base = base
      @dir = dir
      @name = 'index.html'

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'category.html')
      self.data['category'] = category
      self.data['title'] = "카테고리: #{category}"
    end
  end

  class TagPageGenerator < Generator
    safe true

    def generate(site)
      if site.layouts.key? 'tag'
        dir = 'tag'
        site.tags.each_key do |tag|
          site.pages << TagPage.new(site, site.source, File.join(dir, tag), tag)
        end
      end
    end
  end

  class TagPage < Page
    def initialize(site, base, dir, tag)
      @site = site
      @base = base
      @dir = dir
      @name = 'index.html'

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'tag.html')
      self.data['tag'] = tag
      self.data['title'] = "태그: #{tag}"
    end
  end
end
