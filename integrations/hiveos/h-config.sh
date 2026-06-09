#!/usr/bin/env bash

[[ -e /hive/custom ]] && . /hive/custom/keryx-miner/h-manifest.conf
[[ -e /hive/miners/custom ]] && . /hive/miners/custom/keryx-miner/h-manifest.conf

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
