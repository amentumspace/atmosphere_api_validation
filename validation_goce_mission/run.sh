#!/bin/bash
FILES=data/timeseries_data/goce_denswind_ac082_v2_0_20*.txt
for f in $FILES
do
  echo "Processing $f file..."
  python analysis.py --goce_file $f --host $URL
done
