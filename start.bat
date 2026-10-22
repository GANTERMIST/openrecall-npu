@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo Первый запуск: создаю окружение Python 3.12...
  py -3.12 -m venv .venv || (echo Нужен Python 3.12: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install -U pip
  .venv\Scripts\pip install -r requirements.txt || (pause & exit /b 1)
)
start "" .venv\Scripts\pythonw.exe -m winai
