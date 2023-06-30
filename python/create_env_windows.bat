python -m venv ../../env
CALL "..\..\env\Scripts\activate.bat"
pip install -r requirements.txt
start cmd /k CALL "..\..\env\Scripts\activate.bat" /k echo env is ready...
PAUSE
