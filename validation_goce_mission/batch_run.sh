#!/bin/bash

# runs the validation study for a week starting mid-month for 2012
# execute script with $ bash batch_run.sh
# not viable on research plan (requires on-premises or unlimited API server)

for month in 01 02 03 04 05 06 07 08 09 10 11 12
do 
    year="2012"
    day="13"
    date="$year$month$day"

    python analysis.py --hostname http://localhost:5000 \
        --start_date $date --goce_dir ./data/timeseries_data/  
done 


