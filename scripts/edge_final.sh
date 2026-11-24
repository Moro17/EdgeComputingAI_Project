#!/bin/sh
# Final configuration and detection interval N on the i.MX 8M Plus (run on the board from /root/edge; ~10 min).
# Final = fine-tuned YuNet INT8 (uint8 input) on the NPU, lite tracker, dark rule d60_t0.1, look-back 3, fast_blur.
# One run at a time (two NPU processes hung galcore). Copy back: scp -r imx:/root/edge/results experiments/edge/
cd /root/edge
mkdir -p results
V="--video demo.mp4 --loop --seconds 30"
FT=yunet_ft_480x640_u8.tflite
run() { tag=$1; shift; python3 edge_pipeline.py $V --tag $tag --json results/$tag.json "$@" 2>&1 | grep -E "^\{|Traceback|Error"; }

echo "== final configuration: one stream as fast as possible, then live cameras at 25 FPS"
run final_max --model $FT --back 3 --risk d60_t0.1
for k in 1 2 3 4; do run final_cams$k --model $FT --back 3 --risk d60_t0.1 --streams $k --source-fps 25; done

echo "== detection interval N (detector on every N-th frame, boxes held in between)"
for n in 2 3 5; do
  run every${n}_max --model $FT --back 3 --risk d60_t0.1 --every $n
  run every${n}_cams4 --model $FT --back 3 --risk d60_t0.1 --every $n --streams 4 --source-fps 25
done
echo FINAL_DONE
