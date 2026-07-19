#!/usr/bin/env bash
# Compatibility alias — canonical launcher is mmp-launch.sh (the name the mmpOS custom-miner
# spec uses); this exists because field reports show setups looking for "mmp-launcher.sh".
exec "$(dirname "$(realpath "$0")")/mmp-launch.sh" "$@"
