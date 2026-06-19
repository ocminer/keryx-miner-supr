# mmpOS — keryx-miner-supr

Two ways to report keryx-miner-supr stats to mmpOS. Either works; pick one.

## Option A — built-in HTTP endpoint (recommended)
Run the miner with the stats API enabled:
```
keryx-miner-supr -a keryx:<addr>.<worker> -s stratum+tcp://krx.suprnova.cc:4401 \
    --light --api-bind 127.0.0.1:4067
```
Then point the mmpos agent at the endpoint (mmpos agent v4.0.18+ supports a
direct HTTP json endpoint):
```
http://127.0.0.1:4067/mmpos
```
It returns the mmpOS custom-miner JSON directly — no `mmp-stats.sh` needed:
```json
{"busid":[0,1],"hash":[3280000000,190000000],"units":"hs",
 "air":["42","0","0"],"miner_name":"keryx-miner-supr","miner_version":"0.5.1"}
```
(`/` or `/stats` on the same port returns a richer generic JSON for dashboards.)

## Option B — stats script (no miner flag)
Use `keryx-miner-supr/mmp-stats.sh` as the custom-miner stats script. The mmpos
agent calls it as `mmp-stats.sh <DEVICE_NUM> <LOG_FILE>` and it parses the miner
log, emitting the same mmpOS JSON. Drop it next to the miner binary in your mmpos
custom-miner package.

## Notes
- `units` is `hs` (hashes/sec); keryx runs at GH/s scale, so values are ~3.3e9.
- `busid` is the CUDA device index (stable ordinal). `air` = `[accepted, invalid, rejected]`.
- Temps/fans are read by the mmpos agent itself (not reported by the miner).
- keryx is OPoI: it won't mine until model weights are present (`--light` =
  TinyLlama, auto-downloaded on first run) — expect a startup delay before stats.
