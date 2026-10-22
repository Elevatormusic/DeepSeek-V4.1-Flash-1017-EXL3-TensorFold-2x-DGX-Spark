#!/usr/bin/env bash
# hold.sh MARKER: queue on the shared GPU lock of this node and hold it until /tmp/MARKER.run is removed.
# /tmp/MARKER.held exists while the lock is held. The wait limit is 3 h per try.
M=${1:?marker}
touch "/tmp/$M.run"
while ! flock -w 10800 /tmp/spark-gpu.lock bash -c "touch /tmp/$M.held; while [ -e /tmp/$M.run ]; do sleep 3; done; rm -f /tmp/$M.held"; do
  [ -e "/tmp/$M.run" ] || exit 0
done
