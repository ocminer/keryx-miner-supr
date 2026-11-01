#!/usr/bin/env bash
# Regression: H15 AMD rates commonly use uppercase K; treating it as M inflated
# HiveOS hashrates by 1000x. Exercise both shipped callbacks with real log forms.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
for callback in "$root/pkg/keryx-miner-supr/h-stats.sh" "$root/pkg-amd/keryx-miner-supr-amd/h-stats.sh"; do
    eval "$(sed -n '/^to_khs() {/,/^}/p' "$callback")"
    while read -r rate unit expected; do
        for line in "Current hashrate is $rate $unit" "Device #1 AMD Instinct: $rate $unit | 50°C 125.0W | A: 4 / R: 0 | eff 0.002 MH/s/W"; do
            actual=$(printf '%s\n' "$line" | to_khs)
            [[ $actual == "$expected" ]] || { echo "FAIL $callback: $line => $actual != $expected" >&2; exit 1; }
        done
    done <<'CASES'
303.31 Khash/s 303.310
303.31 khash/s 303.310
1.25 Mhash/s 1250.000
1.25 Ghash/s 1250000.000
1.25 Thash/s 1250000000.000
512.5 hash/s 0.512
0 hash/s 0.000
CASES
done
echo 'PASS: both HiveOS callbacks, seven units/rates and two log layouts (28 checks)'
