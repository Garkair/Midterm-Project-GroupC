@echo off
title Social Media Analysis - EDA ^& Visualizations
cls
echo ================================================
echo  STEP 2: EDA ^& VISUALIZATIONS
echo  social_media_analysis.py
echo ================================================
echo.
echo Please wait for computation/execution to finish...
echo Output charts will be saved as PNG files.
echo.
python social_media_analysis.py
echo.
echo ================================================
echo  Done! Check the folder for output PNG files:
echo    - eda_visualizations.png
echo    - model_evaluation.png
echo    - feature_importance.png
echo ================================================
echo.
pause
