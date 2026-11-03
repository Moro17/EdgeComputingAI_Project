#!/bin/sh
# Board measurements for the deliverable README: platform, NPU vs CPU per model, memory, end-to-end CPU vs NPU, load.
cd /root
BM=/usr/bin/tensorflow-lite-2.14.0/examples/benchmark_model
echo "== platform"
uname -r; nproc; cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
grep -E "MemTotal|MemAvailable" /proc/meminfo; df -h / | tail -1
python3 -V; python3 -c "import tflite_runtime,cv2,numpy;print('tflite_runtime',tflite_runtime.__version__,'opencv',cv2.__version__,'numpy',numpy.__version__)"
md5sum yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite
for f in /sys/class/thermal/thermal_zone*/temp; do echo "$f $(cat $f)"; done
echo "== benchmark_model (200 runs)"
for m in yunet_480x640_int8 yunet_ft_480x640_int8 yunet_480x640_fp32 yunet_ft_480x640_fp32 yunet_480x640_fp16 yunet_ft_480x640_fp16 yolov8n-face_320_int8; do
  [ -f $m.tflite ] || continue
  echo "-- $m $(stat -c %s $m.tflite) bytes"
  echo "npu: $($BM --graph=$m.tflite --external_delegate_path=/usr/lib/libvx_delegate.so --num_runs=200 --report_peak_memory_footprint=true 2>&1 | grep -E 'Inference timings|Peak memory|Overall peak' | tr '\n' ' ')"
  for t in 1 4; do
    echo "cpu${t}: $($BM --graph=$m.tflite --num_threads=$t --num_runs=100 --report_peak_memory_footprint=true 2>&1 | grep -E 'Inference timings|Peak memory|Overall peak' | tr '\n' ' ')"
  done
done
echo "== demo end to end, headless, 40 s each"
for mode in npu cpu; do
  for m in yunet_480x640_int8 yunet_ft_480x640_int8; do
    flag=""; [ $mode = cpu ] && flag="--cpu"
    python3 demo.py --video demo.mp4 --loop --display none --switch-every 0 $flag --models $m.tflite > m_demo.log 2>&1 &
    pid=$!; sleep 25
    echo "-- $mode $m: $(grep FPS m_demo.log | tail -5 | tr '\n' '|')"
    echo "   rss_kb $(grep VmRSS /proc/$pid/status | awk '{print $2}') cpu: $(top -b -n 2 -d 5 | grep -E '^ *'$pid' ' | tail -1)"
    echo "   load: $(top -b -n 1 | head -3 | tr '\n' '|')"
    kill $pid; wait $pid 2>/dev/null
  done
done
for f in /sys/class/thermal/thermal_zone*/temp; do echo "$f $(cat $f)"; done
