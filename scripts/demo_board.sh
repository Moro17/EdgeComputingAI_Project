#!/bin/sh
# Demo launcher on the board, output on the HDMI screen. Lives in /root/edge, linked as /usr/bin/demo, so on
# the board (serial console or keyboard) it is just `demo final`. scripts/demo.ps1 types it over the USB serial cable
# (no network needed) or runs it over ssh.
#   demo_board.sh MODE [DATA] [extra edge_pipeline.py args]   start a demo (stops the running one first)
#   DATA: chokepoint | dark (ChokePoint at dark:2) | mv (Music Video) | reflection (door-glass reflections) |
#   nxp (default, demo.mp4);
#   all but nxp come from src/make_demo_videos.py (door also needs the <video>_zone.npy written there)
#   demo_board.sh stop | status | log
# Only one NPU process at a time: two at once hang the galcore driver, so every start stops the previous demo.
cd /root/edge || exit 1
PIDF=/root/edge/demo.pid
LOG=/root/edge/demo.log
export XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1
FT=yunet_ft_480x640_u8.tflite
ST=yunet_480x640_u8.tflite
# HDMI screen of the board (the demo setup); add --display web (browser, port 8080) or both as extra args
DISP=sink

stop() {
    if [ -f "$PIDF" ]; then
        kill "$(cat "$PIDF")" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$(cat "$PIDF")" 2>/dev/null || break; sleep 0.5; done
        kill -9 "$(cat "$PIDF")" 2>/dev/null
        rm -f "$PIDF"
    fi
}

start() {  # $1 = label, rest = command
    label=$1; shift
    stop
    echo "== $label ($(date '+%H:%M:%S'))" > "$LOG"
    nohup "$@" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    sleep 4
    if kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "running: $label (pid $(cat "$PIDF"))"
    else echo "FAILED: $label"; tail -n 15 "$LOG"; rm -f "$PIDF"; exit 1; fi
}

mode=${1:-help}; [ $# -gt 0 ] && shift
VIDEO=demo.mp4
case "$1" in
    chokepoint|cp) VIDEO=chokepoint.mp4; shift ;;
    dark)          VIDEO=chokepoint_dark.mp4; shift ;;
    mv|musicvideo) VIDEO=musicvideo.mp4; shift ;;
    reflection|refl) VIDEO=reflection.mp4; shift ;;
    nxp)           shift ;;
esac
[ -f "$VIDEO" ] || { echo "FAILED: $VIDEO is not on the board (scripts/demo.ps1 install-videos)"; exit 1; }
LIVE="--video $VIDEO --loop --seconds 100000 --source-fps 25 --display $DISP"
case "$mode" in final|door|cams3|cams4|baseline|stock|models)
    case " $* " in *" --display "*) ;; *)
        grep -qx connected /sys/class/drm/card*-HDMI-A-1/status 2>/dev/null ||
            { echo "no HDMI screen: plug in the monitor (or add --display web for the browser)"; exit 1; } ;;
    esac ;;
esac
case "$mode" in
    final)    start "final config: fine-tuned YuNet + tracker + look-back 3 + dark rule, 1 camera" \
                  python3 edge_pipeline.py $LIVE --streams 1 --model $FT --back 3 --risk d60_t0.1 "$@" ;;
    door)     ZONE=${VIDEO%.mp4}_zone.npy
              [ -f "$ZONE" ] || { echo "FAILED: $ZONE missing (door needs a ChokePoint video: chokepoint|reflection)"; exit 1; }
              start "final config + door zone (threshold 0.1 on the door, green outline)"                   python3 edge_pipeline.py $LIVE --streams 1 --model $FT --back 3 --risk d60_t0.1 --zone $ZONE --show-zone "$@" ;;
    cams3)    start "final config on 3 cameras (25 FPS each)" \
                  python3 edge_pipeline.py $LIVE --streams 3 --model $FT --back 3 --risk d60_t0.1 "$@" ;;
    cams4)    start "4 cameras, detection every 2nd frame (N = 2)" \
                  python3 edge_pipeline.py $LIVE --streams 4 --model $FT --back 3 --risk d60_t0.1 --every 2 "$@" ;;
    baseline) start "per-frame baseline: no tracker, no look-back (faces leak)" \
                  python3 edge_pipeline.py $LIVE --streams 1 --model $FT --back 0 --tracker none "$@" ;;
    stock)    start "stock YuNet + tracker + look-back 3 + dark rule" \
                  python3 edge_pipeline.py $LIVE --streams 1 --model $ST --back 3 --risk d60_t0.1 "$@" ;;
    models)   start "Deliverable 1 demo: cycles stock YuNet / fine-tuned YuNet / YOLOv8n-face" \
                  python3 demo.py --video $VIDEO --loop --display $DISP --switch-every 10 \
                  --models yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite ../yolov8n-face_320_int8.tflite "$@" ;;
    stop)     stop; echo "stopped" ;;
    status)   if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then head -n 1 "$LOG"; echo "pid $(cat "$PIDF")"
              else echo "no demo running"; fi ;;
    log)      tail -n 30 "$LOG" ;;
    *)        echo "usage: demo final|door|cams3|cams4|baseline|stock|models [chokepoint|dark|mv|reflection|nxp] [args] | stop | status | log"
              exit 2 ;;
esac
