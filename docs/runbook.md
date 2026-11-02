# Runbook: from the debug cable to the live demo

Board: NXP i.MX 8M Plus EVK, NXP i.MX Release Distro 6.6 (nanbield), user `root` without password.
Everything the demo needs is already in `/root` on the board. To set up a fresh board, see step 6.

## 1. Cables

1. HDMI screen into the board's HDMI port.
2. Micro-USB debug cable from the board's DEBUG port to the Mac.
3. Power on the board. Plug in the HDMI screen before starting the demo: without a screen the video output blocks
   (step 7).

## 2. Serial console

The debug cable shows up as four ports. `-1102` is the Linux console:

```bash
ls /dev/cu.usbserial-*            # expect 1100 1101 1102 1103
screen /dev/cu.usbserial-1102 115200
```

Press Enter. At `imx8mpevk login:` type `root` (no password). If the screen stays blank, try the other three ports.
Leave `screen` with Ctrl-A, then K, then Y.

## 3. Network address

The board joins the Wi-Fi by itself at boot (connman, `/var/lib/connman/home-wifi.config`). On the serial console:

```bash
ip -4 addr show wfd0              # e.g. inet 192.168.1.137/24
```

The address comes from DHCP and changes between boots (it was .135, then .137).

No `inet` line means the Wi-Fi didn't connect:

```bash
connmanctl services               # the network should be listed with *AR (auto, ready)
connmanctl enable wifi; connmanctl scan wifi; connmanctl services
```

For a different Wi-Fi network, add a `[service_...]` block with `Type = wifi`, `Name = <SSID>` and
`Passphrase = <password>` to that config file, then reboot. Avoid `connmanctl connect` in interactive mode: it once left
the Wi-Fi interface down.

## 4. SSH from the Mac

Put the address from step 3 into `~/.ssh/config`:

```
Host imx
    HostName 192.168.1.137
    User root
```

```bash
ssh imx 'hostname'                # imx8mpevk
```

If SSH warns about the host key after an address change, check it is the same board before trusting it. The key
fingerprints must match the old entry:

```bash
ssh-keygen -F 192.168.1.135 -l                          # old address, from known_hosts
ssh-keyscan 192.168.1.137 2>/dev/null | ssh-keygen -lf -  # new address
```

If they match, append the `ssh-keyscan` output to `~/.ssh/known_hosts`.

## 5. Check the board

```bash
ssh imx 'cat /sys/class/drm/*/status; ps | grep [w]eston | head -2; ls /root/*.tflite /root/demo.mp4'
```

- `connected`: the HDMI screen is seen. `disconnected` means check the cable or the screen input.
- Weston processes are listed: it starts at boot.
- `yunet_480x640_int8.tflite`, `yunet_ft_480x640_int8.tflite`, `yolov8n-face_320_int8.tflite` and `demo.mp4` exist.

## 6. Copy files (fresh board, or after code changes)

From the repo root on the Mac:

```bash
scp src/demo.py src/board_bench.py src/systems/anonymize.py scripts/run_demo.sh imx:/root/
scp models/yunet_480x640_int8.tflite models/yunet_ft_480x640_int8.tflite models/yolov8n-face_320_int8.tflite imx:/root/
ssh imx 'chmod +x /root/run_demo.sh'
```

`demo.mp4` is an H.264 800x600 clip of the test sequences P2E_S1_C1.1, P2E_S1_C2.1 and P2L_S1_C1.1.
The model files and the clip are not in git.

## 7. Run the demo

```bash
ssh imx 'cd /root; pkill -f "python3 [d]emo"; nohup ./run_demo.sh --video demo.mp4 --loop \
  --models yunet_480x640_int8.tflite yunet_ft_480x640_int8.tflite yolov8n-face_320_int8.tflite \
  > demo.log 2>&1 < /dev/null &'
```

The screen shows the anonymized video, with the model name, FPS and face count in the overlay. The model changes
every 10 s (`--switch-every N`). Operating thresholds: stock 0.6, fine-tuned 0.7, YOLO 0.49. Override them with
`--thr`, one value per model.

Watch the FPS from the Mac (one line every 2 s):

```bash
ssh imx 'grep FPS /root/demo.log | tail -5'
```

Expected on HDMI: stock YuNet about 28.5 FPS, fine-tuned YuNet about 27.9, YOLOv8n-face 320 about 22.1.

Stop it:

```bash
ssh imx 'pkill -f "python3 [d]emo"'
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No `FPS` lines in `demo.log`, black screen | No HDMI screen seen (`disconnected` in step 5). Connect the screen, restart the demo. To measure without a screen: `--display none`. |
| `demo.py: cannot open video ...` | Wrong `--video` path; the demo now exits instead of spinning. |
| FPS around 16 | Started with `--display window` (`cv2.imshow` costs 25 ms per frame). Use the default `sink`. |
| Demo hangs at start with a fullscreen sink | `fullscreen=true` hangs waylandsink; the demo scales to 1440x1080 instead. Don't add it. |
| `ssh: connect ... timed out` | Board address changed: repeat step 3 on the serial console, update `~/.ssh/config`. |
| `pkill` over SSH kills the SSH session | The pattern matches its own command line; keep the `[d]emo` bracket form. |
