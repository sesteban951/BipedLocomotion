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

num_trials=10
num_falls=0

# Run the experiment
for i in $(seq 1 $num_trials); do
    echo "***Running trial $i/$num_trials ***"
    python3 mpc.py > /dev/null
    if [ $? -ne 0 ]; then
        num_falls=$((num_falls+1))
    fi
done

echo "Over $num_trials trials, the robot fell $num_falls times"