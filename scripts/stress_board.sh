#!/bin/sh
# Stress tier on the NPU (run on the board from /root, needs board_bench.py + degrade.py + the .tflite models).
# Resumable: finished sequences are skipped, so after a reboot just start it again.
#   setsid nohup ./stress_board.sh "dark:2 blur:2 down:2 occl:2 jpeg:2" > stress.log 2>&1 < /dev/null &
DEGS=${1:-"dark:2 blur:2 down:2 occl:2 jpeg:2"}
MODELS=${2:-"yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite"}
cd /root
for d in $DEGS; do
  for m in $MODELS; do
    python3 board_bench.py --model $m --frames /root/chokepoint --out /root/dets --degrade $d
  done
done
echo STRESS_DONE
