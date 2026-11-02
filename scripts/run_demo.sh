#!/bin/sh
# Board launcher: sets the Wayland env and runs demo.py from /root. Args are passed to demo.py.
cd /root
XDG_RUNTIME_DIR=/run/user/0 WAYLAND_DISPLAY=wayland-1 exec python3 demo.py "$@"
