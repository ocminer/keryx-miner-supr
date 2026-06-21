# RTX 5090 tuning recipe — ~+8-9% KeryxHash for free

On KeryxHash the RTX 5090 is **power-bound**, not compute-bound: at its stock 575 W
limit the core throttles itself *down* (≈2700 MHz) to stay inside the power budget,
and hashrate scales almost linearly with the watts you give it. That means the win
isn't a different kernel — it's **more useful work per watt**, via a higher power
limit plus a positive **core-clock offset** (an effective undervolt: the same voltage
now buys a higher clock, so more of the 600 W turns into hashing).

Measured on a stock RTX 5090 (driver 580), byte-exact (identical hash output — verified
with the miner's correctness checksum, so **zero extra rejects**):

| Power limit | Core offset | Hashrate | vs stock |
|------------:|------------:|---------:|---------:|
| 575 W (stock) | +0 | ~3.24 GH/s | — |
| 600 W | +0 | ~3.33 GH/s | +2.7% |
| **600 W** | **+250 MHz** | **~3.50 GH/s** | **+8%** |
| 600 W | +300 MHz | ~3.54 GH/s | +9% |

## Recommended production setting

**Power limit 600 W, core-clock offset +250 MHz.** (Memory clock: leave stock — KeryxHash
uses almost no memory bandwidth, but lowering it doesn't help here and can starve the card.)

+250 MHz is a deliberately safe margin. Higher offsets keep gaining a little, but past
roughly +350-400 MHz the GPU's firmware can crash and require a reset, so **don't chase
the last MHz** — +250 captures almost all of the benefit with headroom for hot summer
days and silicon variation. Always confirm shares still get **accepted** (0 new rejects)
after applying an offset; if rejects appear, lower the offset.

## How to apply it

Use whatever overclocking tool your setup already has — the *values* are what matter:

- **HiveOS / mmpOS (overclocking tab or flight-sheet OC):**
  `Core Clock: 250`, `Power Limit: 600` (memory left at 0/stock).
- **Windows (MSI Afterburner):** Power Limit `100%` (= 600 W), Core Clock `+250`.
  Leave memory at 0.
- **Linux desktop (X session):**
  `nvidia-settings -a [gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=250`
  plus `sudo nvidia-smi -pl 600`.

> Note: the raised power limit is **not persistent** — it resets to 575 W on reboot
> unless your OS re-applies it (HiveOS/mmpOS do this for you; on bare Linux re-run
> `nvidia-smi -pl 600` at boot).

### suprtuner (headless, coming soon)

For **headless Linux rigs without an X server**, `nvidia-smi` alone can't set a clock
offset. We're releasing a small companion tool, **suprtuner**, that applies the core/mem
offset, power limit and locked clocks directly through NVML (no X, no reboot) and stays
resident to re-apply through driver/P-state events:

```
sudo suprtuner --devices 0 --core-offset 250 --power 600 --interval 5
```

(Published separately once it's out — until then use your platform's OC tool above.)

## Why this is the only lever

KeryxHash is two Keccak permutations plus a small matmul — the kernel is already at its
energy-per-hash floor and the GPU runs at 100% during hashing. When you're power-bound,
nothing in the kernel can make it faster without drawing more watts; the gain comes from
running the watts you have at a more efficient point on the voltage/frequency curve.
That's what the offset does, and ~+8-9% is what's on the table.
