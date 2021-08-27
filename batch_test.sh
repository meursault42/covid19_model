# step one: retrieve and process county level JHU data, combine with other data sources
Rscript --1_county_agg_code.R
wait

::step two: retrieve and process 
Rscript --2_state_agg_code.R
wait

::step three: process and smooth data for downstream processing
python 3_gen_smooths.py
wait

::step four: run ts model
python 4_ts_loop_code.py

::step five: run XGB model
python 5_xgb_model.py

::step six: reformat model output
wait
python 6_model_output_reformatter.py
