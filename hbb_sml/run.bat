@echo off
chcp 65001 >nul
D:\Miniconda3\python.exe "%~dp0downsample_and_pad.py" %*
