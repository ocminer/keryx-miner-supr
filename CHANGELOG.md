# keryx-miner-supr 0.14.7

Faster AMD mining, with mining and inference continuing on the same card. This release improves RX 7900 XTX and Instinct MI50 performance and fixes device selection on mixed AMD rigs.

## Measured AMD performance

Controlled Linux measurements with the same model and power limit in each comparison:

| GPU | Test | Power limit | 0.14.6 | Optimized build | Gain |
|---|---|---:|---:|---:|---:|
| Radeon RX 7900 XTX | GPU benchmark | 244 W | 1.450 MH/s | 1.535 MH/s | +5.87% |
| Radeon RX 7900 XTX | Pool mining | 219 W | 1.248 MH/s | 1.285 MH/s | +2.93% |
| Instinct MI50, sample 1 | Pool mining | 100 W | 0.419 MH/s | 0.544 MH/s | +29.69% |
| Instinct MI50, sample 2 | Pool mining | 100 W | 0.426 MH/s | 0.551 MH/s | +29.23% |
| Radeon RX 7600 XT | Pool mining | 165 W | 0.590 MH/s | 0.590 MH/s | unchanged |

These optimization measurements preceded release packaging. Pool figures compare the final 70 seconds of matched 180-second runs. The RX 7900 XTX and RX 7600 XT also served inference requests while mining. Results depend on the model, cooling, power limit, driver and inference activity; short tests do not establish sustained full-power performance. The measured gains apply to the tested Linux/RADV configuration.

## Downloads

| Platform | Package |
|---|---|
| NVIDIA Turing and newer, Linux, driver 575+ | `modern`: standalone, HiveOS, mmpos, SMOS |
| NVIDIA Volta/Pascal, older CMP cards, or driver 550–574 | `legacy`: standalone, HiveOS, mmpos, SMOS |
| AMD, Linux | `amd`: standalone, HiveOS, mmpos |
| NVIDIA, Windows | `keryx-miner-supr-windows-nvidia-pom.zip` |
| AMD, Windows | `keryx-miner-supr-windows-amd.zip` |
| Apple Silicon, macOS | `keryx-miner-supr-macos-arm64-0.14.7.tar.gz` |
| Docker, NVIDIA modern | `ocminersupr/keryx-miner-supr:0.14.7` |

Use the matching SHA256SUMS file to verify downloads. Existing pool, wallet and model settings can be retained. No protocol or model changes are required.
