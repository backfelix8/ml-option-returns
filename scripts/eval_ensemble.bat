setlocal

set SCRIPT=evaluate.py
set OUTFILE=evaluate_ensemble.out

echo Starte Jobs... > "%OUTFILE%"

py -3.12 -u %SCRIPT% notused shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% notused robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% notused performance >> "%OUTFILE%" 2>&1

git add ensemble_files
git commit -m "Update ensemble eval"
git push

echo Fertig! >> "%OUTFILE%"