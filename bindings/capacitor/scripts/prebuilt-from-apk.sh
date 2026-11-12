#!/bin/sh
# The ready-made native library for the npm package, taken from an app that was built from source with this
# plugin (so it is the library that app ran with). Usage: sh scripts/prebuilt-from-apk.sh path/to/app.apk
set -e
apk="$1"
[ -f "$apk" ] || { echo "usage: $0 app.apk"; exit 1; }
out="$(dirname "$0")/../android/prebuilt/arm64-v8a"
rm -rf "$out" && mkdir -p "$out"
unzip -j -o -q "$apk" "lib/arm64-v8a/libkneejerk_jni.so" "lib/arm64-v8a/libc++_shared.so" -d "$out"
if strings -n 6 "$out/libkneejerk_jni.so" | grep -q "/home/\|/Users/"; then echo "build-machine paths in the library, not packaging it"; exit 2; fi
ls -la "$out"
