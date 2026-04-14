@echo off
title Social Media Analysis - Data Cleaning
cls
echo ================================================
echo  STEP 1: DATA CLEANING
echo  Cleaning.py
echo ================================================
echo.
echo Please wait for computation/execution to finish...
echo.
cd C:\Users\jd\Desktop\DBU\cleanup_py_midt\v3
python Cleaning.py
echo.
echo ================================================
echo  Done! Output saved to:
echo    - cleaned_social_media_productivity.csv
echo ================================================
echo.
pause
