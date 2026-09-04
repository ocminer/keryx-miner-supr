#!/usr/bin/env bash

cd `dirname $0`

[ -t 1 ] && . colors

. h-manifest.conf

[[ -z $CUSTOM_LOG_BASENAME ]] && echo -e "${RED}No CUSTOM_LOG_BASENAME is set${NOCOLOR}" && exit 1
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && exit 1
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}Custom config ${YELLOW}$CUSTOM_CONFIG_FILENAME${RED} is not found${NOCOLOR}" && exit 1

# Expose the miner dir and CUDA runtime libs (cuBLAS etc.) for OPoI inference.
# The binary self-installs cuBLAS on first run and registers its path with ldconfig,
# so this is only a belt-and-suspenders hint for the dynamic loader.
export LD_LIBRARY_PATH="$(dirname $0):${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"

CLI_ARGS="$(< "$CUSTOM_CONFIG_FILENAME") $*"
case " $CLI_ARGS " in
  *" --no-tui "*) ;;
  *) CLI_ARGS="--no-tui $CLI_ARGS" ;;
esac
# Intentional word splitting: HiveOS stores the complete miner command line in this config file.
# shellcheck disable=SC2086
./$CUSTOM_MINERBIN $CLI_ARGS 2>&1 | tee "$CUSTOM_LOG_BASENAME.log"
