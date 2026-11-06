#!/usr/bin/env bash
# Check a kernel kit folder against its MANIFEST. This verifier belongs to this repo. It never runs a file from the kit.
#   scripts/verify_kit.sh KIT_DIR                  (on this node)
#   bash -s -- KIT_DIR < scripts/verify_kit.sh     (on the other node: the script goes in on stdin)
# The kit is correct only when all of these are true:
#   - MANIFEST exists, and each line is "sha256, two spaces, a relative path" (no absolute path, no "..");
#   - every file that MANIFEST lists exists with the sha256 that MANIFEST gives;
#   - the folder holds no file that MANIFEST does not list (MANIFEST itself is the only exception);
#   - the folder holds no symbolic link and no other special file.
# Exit 0: the kit is correct. Exit 1: it is not (each problem is printed). Exit 2: wrong use.
set -u
if [ $# -ne 1 ] || [ ! -d "$1" ]; then
  echo "usage: verify_kit.sh KIT_DIR" >&2
  exit 2
fi
cd "$1" || exit 2
bad=0
problem() { echo "kit NOT OK: $*"; bad=1; }

[ -f MANIFEST ] && [ ! -L MANIFEST ] || { echo "kit NOT OK: MANIFEST is missing"; exit 1; }

# 1. The form of each MANIFEST line.
[ -s MANIFEST ] || problem "MANIFEST is empty"
if grep -qvE '^[0-9a-f]{64}  [^/ ][^ ]*$' MANIFEST; then
  problem "MANIFEST has a line that is not 'sha256, two spaces, path'"
fi
if grep -qE '(^|/)\.\.(/|$)|(^|  )/' MANIFEST; then
  problem "MANIFEST lists a path with .. or a path that starts with /"
fi
[ $bad = 0 ] || exit 1

# 2. Links and special files.
special=$(find . ! -type f ! -type d | head -n 5)
[ -z "$special" ] || problem "the folder holds a link or a special file: $(echo "$special" | tr '\n' ' ')"

# 3. The file list of the folder is the file list of MANIFEST.
listed=$(sed 's/^[0-9a-f]\{64\}  //' MANIFEST | LC_ALL=C sort -u)
found=$(find . -type f ! -path ./MANIFEST | sed 's|^\./||' | LC_ALL=C sort)
extra=$(LC_ALL=C comm -13 <(echo "$listed") <(echo "$found"))
missing=$(LC_ALL=C comm -23 <(echo "$listed") <(echo "$found"))
[ -z "$extra" ] || problem "a file is not in MANIFEST: $(echo "$extra" | head -n 5 | tr '\n' ' ')"
[ -z "$missing" ] || problem "a file of MANIFEST is missing: $(echo "$missing" | head -n 5 | tr '\n' ' ')"

# 4. The sha256 of every listed file. Skip this when the file list is wrong (sha256sum would only repeat the error).
if [ $bad = 0 ]; then
  sha256sum -c --strict --quiet MANIFEST > /dev/null 2>&1 || {
    problem "a file differs from MANIFEST: $(sha256sum -c --strict MANIFEST 2>&1 | grep -v ': OK$' | head -n 5 | tr '\n' ' ')"
  }
fi

if [ $bad = 0 ]; then
  echo "kit OK: $(wc -l < MANIFEST | tr -d ' ') files match MANIFEST"
  exit 0
fi
exit 1
