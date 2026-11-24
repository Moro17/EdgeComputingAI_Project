"""TFLite detector latency on the Jetson Nano CPU (4x Cortex-A57, 4 threads), the same .tflite files as the i.MX board.

tflite_runtime 2.5 from Google's Coral index, installed with --no-deps (JetPack 4.6 numpy 1.13):
    pip3 install --user --no-deps --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
    python3 jetson_tflite_bench.py yunet_480x640_fp32.tflite yunet_480x640_int8.tflite  # -> tfl_cpu.json
"""
import sys, time, json
import numpy as np
import tflite_runtime.interpreter as tfl
res = {}
for path in sys.argv[1:]:
    for th in (4,):
        it = tfl.Interpreter(model_path=path, num_threads=th)
        it.allocate_tensors()
        d = it.get_input_details()[0]
        x = (np.random.rand(*d["shape"]) * 255).astype(d["dtype"]) if d["dtype"] != np.float32 else (np.random.rand(*d["shape"]) * 255).astype(np.float32)
        it.set_tensor(d["index"], x)
        for _ in range(5): it.invoke()
        t = []
        for _ in range(40):
            t0 = time.perf_counter(); it.invoke(); t.append(1000 * (time.perf_counter() - t0))
        res["%s_t%d" % (path, th)] = round(float(np.median(t)), 1)
        print(path, th, res["%s_t%d" % (path, th)], str(d["dtype"]), list(d["shape"]), flush=True)
json.dump(res, open("tfl_cpu.json", "w"), indent=1)
