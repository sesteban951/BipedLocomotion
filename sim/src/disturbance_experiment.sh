#!/bin/bash

##
#
# Bash script to measure the disturbance-rejection abilities of the robot.
# 
# All this script really does is run `mpc.py` N times and record whether it
# exits successfully or not. The script failing indicates that the robot fell.
#
# Before running this script, verify in config.yaml that:
#   - The fixed random seed is disabled
#   - You have selected the correct controller
#   - Disturbance parameters are set correctly
#   - The overall simulation duration is set to something reasonably small
#
##


num_trials=500
num_falls=0

# Set up a file to save the results
save_file="data/data_random_disturbance_mpc.csv"
echo "fx, fy, fz, fell" > $save_file

# Run the experiment
for i in $(seq 1 $num_trials); do
    echo "***Running trial $i/$num_trials ***"

    # Run mpc.py and grab the disturbance values by matching the pattern
    # Disturbance vector:  [-114.04944531  122.2075357  -231.45198683].
    output=$(python3 mpc.py)
    fell=$?

    dist=$(echo $output | grep -oP '\[\K[^\]]+')
    d1=$(echo $dist | cut -d' ' -f1)
    d2=$(echo $dist | cut -d' ' -f2)
    d3=$(echo $dist | cut -d' ' -f3)

    if [ $fell -ne 0 ]; then
        num_falls=$((num_falls+1))
    fi

    # Update the file
    echo "$d1, $d2, $d3, $fell" >> $save_file

done

echo "Over $num_trials trials, the robot fell $num_falls times"