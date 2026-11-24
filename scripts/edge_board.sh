#!/bin/sh
# Edge pipeline measurements on the i.MX 8M Plus (run on the board from /root/edge; ~15 min).
# Needs: edge_pipeline.py board_bench.py anonymize.py bytelite.py demo.mp4, yunet{,_ft}_480x640_{int8,u8,fp32}.tflite
# (the _u8 files come from src/export_uint8_input.py). One run at a time: two processes on the NPU hung galcore.
# Writes results/*.json (one per configuration) and results/accelerators.txt. Copy back:
#   scp -r imx:/root/edge/results experiments/edge/
cd /root/edge
mkdir -p results
V="--video demo.mp4 --loop --seconds 30"
FT=yunet_ft_480x640
run() { tag=$1; shift; python3 edge_pipeline.py $V --tag $tag --json results/$tag.json "$@" 2>&1 | grep -E "^\{|Traceback|Error"; }

echo "== optimization steps, one stream as fast as possible"
run s1_serial_float_cpuresize --serial --tracker none --g2d-resize 0 --back 0 --model ${FT}_int8.tflite
run s2_serial_u8_cpuresize    --serial --tracker none --g2d-resize 0 --back 0 --model ${FT}_u8.tflite
run s3_serial_u8_g2d          --serial --tracker none --g2d-resize 1 --back 0 --model ${FT}_u8.tflite
run s4_pipelined_u8_g2d       --tracker none --g2d-resize 1 --back 0 --model ${FT}_u8.tflite
run s5_pipelined_lite_back3   --g2d-resize 1 --back 3 --model ${FT}_u8.tflite

echo "== live cameras at 25 FPS, privacy policy lite tracker + back3 (default)"
for k in 1 2 3 4; do run cams${k}_ft --streams $k --source-fps 25 --back 3 --model ${FT}_u8.tflite; done
run cams4_stock --streams 4 --source-fps 25 --back 3 --model yunet_480x640_u8.tflite
run cams4_ft_npu+cpu --streams 4 --source-fps 25 --back 3 --model ${FT}_u8.tflite --cpu-model ${FT}_fp32.tflite --cpu-threads 2

echo "== accelerators: the same network on NPU, 3D GPU and CPU (benchmark_model)"
BM=/usr/bin/tensorflow-lite-2.14.0/examples/benchmark_model
{
  cat /sys/kernel/debug/gc/clk 2>/dev/null
  for m in ${FT}_u8 ${FT}_int8 ${FT}_fp32; do
    echo "-- $m"
    echo "npu (vx delegate): $($BM --graph=$m.tflite --external_delegate_path=/usr/lib/libvx_delegate.so --num_runs=100 2>&1 | grep 'Inference timings' | tail -1)"
    echo "gpu3d (vx, USE_GPU_INFERENCE=1): $(USE_GPU_INFERENCE=1 $BM --graph=$m.tflite --external_delegate_path=/usr/lib/libvx_delegate.so --num_runs=20 2>&1 | grep 'Inference timings' | tail -1)"
    echo "gpu3d (opencl delegate): $($BM --graph=$m.tflite --use_gpu=true --num_runs=20 2>&1 | grep 'Inference timings' | tail -1)"
    echo "cpu 4 threads (xnnpack): $($BM --graph=$m.tflite --num_threads=4 --num_runs=50 2>&1 | grep 'Inference timings' | tail -1)"
  done
} > results/accelerators.txt 2>&1
cat results/accelerators.txt
echo EDGE_DONE
