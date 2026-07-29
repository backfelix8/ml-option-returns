setlocal

set SCRIPT=evaluate.py
set OUTFILE=evaluate_all_performance.out

echo Starte Jobs... > "%OUTFILE%"

py -3.12 -u %SCRIPT% gbrt_standard performance >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_discrete performance >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_l2 performance >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn performance >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% gbrt_standard_l2 performance >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% rf_standard_l2 performance >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_noise performance >> "%OUTFILE%" 2>&1


echo Fertig! >> "%OUTFILE%"