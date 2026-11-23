#!/bin/bash
# NPU detections on the Music Video dataset (src/loaders/musicvideo.py), run from the PC, like val_board.sh:
# one sequence (half a video, ~0.4 GB of JPEG) at a time is copied to /root/mv, detected with every model, deleted.
# Resumable (board_bench.py skips finished sequences). Detections land in /root/dets_mv/<system>[@deg]/.
#   ECA_DATA=/path/to/data bash scripts/mv_board.sh [ssh-host] [KIND:SEV]
# Afterwards: ssh imx 'cd /root/dets_mv && tar czf - .' | tar xzf - -C <benchmark --out dir>/dets
set -e
HOST=${1:-imx}
ssh() { command ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$@"; }  # a dropped link killed a run
DEG=${2:-}
SFX=${DEG:+@${DEG/:/}}
DARG=${DEG:+--degrade $DEG}
FR=${ECA_DATA:?set ECA_DATA}/musicvideo/frames
MODELS="yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite"
for s in $(cd "$FR" && ls -d */ | tr -d /); do
  todo=0
  for n in yunet_480x640_npu yunet_ft_480x640_npu; do
    ssh $HOST "test -f /root/dets_mv/$n$SFX/$s.json" || todo=1
  done
  [ $todo = 0 ] && { echo "skip $s"; continue; }
  echo "copy $s"
  ssh $HOST "rm -rf /root/mv && mkdir -p /root/mv"
  (cd "$FR" && tar cf - $s) | ssh $HOST "tar xf - -C /root/mv"
  for m in $MODELS; do
    ssh $HOST "cd /root && python3 board_bench.py --model $m --frames /root/mv --out /root/dets_mv $DARG"
  done
  ssh $HOST "rm -rf /root/mv"
done
echo MV_DONE
