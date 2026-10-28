@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 母婴健康 AI 答疑助手
echo ==============================================
echo   母婴健康 AI 答疑助手 - 正在启动...
echo   首次启动会自动打开浏览器
echo ==============================================
echo.
python -m streamlit run app.py
echo.
echo 程序已退出，按任意键关闭窗口...
pause >nul
