#!/bin/bash
# Clean NPU detections on the ChokePoint val split (P2E/P2L S2-S4), run from the PC.
# The board's root partition cannot hold all val frames (4.9 GB), so each archive (6 sequences) is copied to
# /root/val, detected with every model, deleted, then the next one. Resumable: board_bench.py skips finished
# sequences, and archives whose sequences are all done are not copied again.
#   ECA_DATA=/path/to/data bash scripts/val_board.sh [ssh-host] [KIND:SEV]
# Afterwards: ssh imx 'cd /root && tar czf - dets' | tar xzf - -C <benchmark --out dir>
set -e
HOST=${1:-imx}
DEG=${2:-}
SFX=${DEG:+@${DEG/:/}}
DARG=${DEG:+--degrade $DEG}
FR=${ECA_DATA:?set ECA_DATA}/chokepoint/frames
MODELS="yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite"
for a in P2E_S2 P2E_S3 P2E_S4 P2L_S2 P2L_S3 P2L_S4; do
  seqs=$(cd "$FR" && ls -d ${a}_*)
  todo=0
  for s in $seqs; do
    for n in yunet_480x640_npu yunet_ft_480x640_npu; do
      ssh $HOST "test -f /root/dets/$n$SFX/$s.json" || todo=1
    done
  done
  [ $todo = 0 ] && { echo "skip $a"; continue; }
  echo "copy $a"
  ssh $HOST "rm -rf /root/val && mkdir -p /root/val"
  (cd "$FR" && tar cf - $seqs) | ssh $HOST "tar xf - -C /root/val"
  for m in $MODELS; do
    ssh $HOST "cd /root && python3 board_bench.py --model $m --frames /root/val --out /root/dets $DARG"
  done
  ssh $HOST "rm -rf /root/val"
done
echo VAL_DONE
