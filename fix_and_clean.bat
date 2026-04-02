@echo off
echo ================================================
echo  Cleaning old pages and fixing dependencies
echo ================================================

:: Delete all old pages with = in the name
echo Deleting old pages with = in filename...
del /f /q "pages\1=Data_Upload.py"      2>nul
del /f /q "pages\2=Model_Training.py"   2>nul
del /f /q "pages\3=Forecasting.py"      2>nul
del /f /q "pages\4=Model_Comparison.py" 2>nul
del /f /q "pages\5=ARIMA_Analysis.py"   2>nul
del /f /q "pages\6=Run_History.py"      2>nul

:: Also delete any emoji-named leftover files
del /f /q "pages\1_*_Data_Upload.py"      2>nul
del /f /q "pages\2_*_Model_Training.py"   2>nul
del /f /q "pages\3_*_Forecasting.py"      2>nul
del /f /q "pages\4_*_Model_Comparison.py" 2>nul
del /f /q "pages\5_*_ARIMA_Analysis.py"   2>nul
del /f /q "pages\6_*_Run_History.py"      2>nul

echo.
echo Remaining pages:
dir /b pages\

echo.
echo ================================================
echo  Installing missing packages...
echo ================================================
python -m pip install plotly statsmodels

echo.
echo ================================================
echo  Done! Now run:  streamlit run app.py
echo ================================================
pause
