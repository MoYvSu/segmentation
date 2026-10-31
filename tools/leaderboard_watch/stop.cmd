@echo off
python "%~dp0app.py" stop
if errorlevel 1 pause
