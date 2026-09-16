setlocal

set SCRIPT=evaluate.py
set OUTFILE=evaluate_all.out

echo Starte Jobs... > "%OUTFILE%"

py -3.12 -u %SCRIPT% ffn shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_discrete robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_discrete shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_l2 robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_l2 shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% gbrt_standard robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% gbrt_standard shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% gbrt_standard_l2 robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% gbrt_standard_l2 shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% rf_standard robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% rf_standard shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% rf_standard_l2 robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% rf_standard_l2 shapley >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_noise robustness >> "%OUTFILE%" 2>&1
py -3.12 -u %SCRIPT% ffn_noise shapley >> "%OUTFILE%" 2>&1


echo Fertig! >> "%OUTFILE%"