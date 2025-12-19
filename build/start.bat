@echo off
echo ========================================
echo ZtoD Blog Auto Start Script
echo ========================================
echo.

echo [1/3] Generating category and tag pages...
python generate_category_tag_pages.py
if %errorlevel% neq 0 (
    echo ERROR: Python script failed!
    pause
    exit /b %errorlevel%
)
echo.

echo [2/3] Starting Jekyll build...
bundle exec jekyll build
if %errorlevel% neq 0 (
    echo ERROR: Jekyll build failed!
    pause
    exit /b %errorlevel%
)
echo.

echo [3/3] Starting Jekyll server...
echo.
echo Server is running!
echo URL: http://localhost:4000/zero_to_deep_kh
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.
bundle exec jekyll serve

pause
