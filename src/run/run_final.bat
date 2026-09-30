@echo off
REM Team zippy - full local pipeline for the final submission (Windows).  Run from code\business_entity_resolution.
REM Cross-encoder score files (ce1, ce2, ce4) must be in %KOUT% (trained/scored on Kaggle, see README 5.3).
setlocal
set ROOT=C:\projects\AmazonML
set DATA=%ROOT%\student_resource\dataset
set WORK=%ROOT%\work3
set KOUT=%ROOT%\kaggle_out
set OUT_V7=%ROOT%\output_v7
set OUT=%ROOT%\output_final
if not exist logs mkdir logs
python src\prepare.py --data %DATA% --work %WORK% --workers 10 > logs\1_prepare.log 2>&1 || goto :err
python src\block.py --work %WORK% --split train --workers 6 > logs\2_block_train.log 2>&1 || goto :err
python src\block.py --work %WORK% --split test --workers 6 > logs\3_block_test.log 2>&1 || goto :err
python src\block2.py pool --work %WORK% --split train --workers 6 > logs\4_pool_train.log 2>&1 || goto :err
python src\block2.py pool --work %WORK% --split test --workers 6 > logs\5_pool_test.log 2>&1 || goto :err
python src\block2.py fit --work %WORK% --data %DATA% --workers 10 > logs\6_fit.log 2>&1 || goto :err
python src\block2.py apply --work %WORK% --data %DATA% --split train --workers 10 --budget 10 > logs\7_apply_train.log 2>&1 || goto :err
python src\block2.py apply --work %WORK% --data %DATA% --split test --workers 10 --budget 10 > logs\8_apply_test.log 2>&1 || goto :err
python src\train.py --work %WORK% --data %DATA% --workers 10 --blocks blocks_train_v7 --K 1000 --side 0 --loco 1 --iters 3000 > logs\9_train.log 2>&1 || goto :err
python src\predict.py --work %WORK% --data %DATA% --out %OUT_V7% --workers 8 --blocks blocks_test_v7 > logs\10_predict.log 2>&1 || goto :err
python src\ce_score_local.py --work %WORK% --ce_dir %KOUT% > logs\11_ce_local.log 2>&1 || goto :err
echo 0.5> %KOUT%\blend_w.txt
python src\blend_ce.py --work %WORK% --art %WORK% --data %DATA% --ce_dir %KOUT% --out %OUT% --tags ce1,ce2,ce4 > logs\12_blend.log 2>&1 || goto :err
copy /Y %OUT_V7%\candidate_pairs.tsv %OUT%\candidate_pairs.tsv >nul
python %ROOT%\student_resource\utils\validate_submission.py --matching %OUT%\matching_results.tsv --candidate %OUT%\candidate_pairs.tsv --test-dir %DATA%\test
echo DONE: %OUT%
goto :eof
:err
echo FAILED - see logs
