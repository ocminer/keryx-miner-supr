#!/usr/bin/env bash
# Package one build "line" (legacy or modern) into all distribution formats, using
# the binaries + CUDA runtime libs already produced in hiveos/<DISTDIR>/ by
# build-offline.sh. The bundled libs ARE the line's CUDA version (legacy=12.2/
# floor535, modern=12.9/floor575), so each package carries its own driver floor.
#
# HiveOS/SMOS: the line is embedded in the MINER NAME (keryx-miner-supr-<line>) so
# HiveOS parses <name>-<version>.tar.gz correctly (it splits on the LAST '-' as the
# version) and the unpacked top-folder + CUSTOM_NAME + hardcoded /hive paths all
# match. A "-<line>" suffix AFTER the version breaks that parse.
#
# Usage: package-line.sh <DISTDIR> <LABEL>   e.g. package-line.sh dist-legacy legacy
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISTDIR="$1"; LABEL="$2"
D="$REPO/hiveos/$DISTDIR"
HPKG="$REPO/hiveos/pkg/keryx-miner-supr"
MPKG="$REPO/mmpos/keryx-miner-supr"
NAME=keryx-miner-supr
VER=$(grep -m1 '^CUSTOM_VERSION=' "$HPKG/h-manifest.conf" | cut -d= -f2)

[[ -f "$D/keryx-miner-supr" ]] || { echo "ERROR: $D/keryx-miner-supr (static) missing"; exit 1; }
[[ -f "$D/keryx-miner-supr-dynamic" ]] || { echo "ERROR: $D/keryx-miner-supr-dynamic missing"; exit 1; }
LIBS=(libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10)

mklib(){ mkdir -p "$1/lib"; for l in "${LIBS[@]}"; do cp -L "$D/lib/$l" "$1/lib/"; done; }
# Rewrite the HiveOS miner-dir name in the hardcoded h-* paths + CUSTOM_NAME (the
# binary stays ./keryx-miner-supr; only the /hive + /var/log dir component changes).
hrename(){ # $1=dir of h-* files  $2=new miner name
  sed -i -e "s|^CUSTOM_NAME=.*|CUSTOM_NAME=$2|" \
         -e "s|/hive/miners/custom/keryx-miner-supr/|/hive/miners/custom/$2/|g" \
         -e "s|/var/log/miner/keryx-miner-supr/|/var/log/miner/$2/|g" \
         "$1"/h-manifest.conf "$1"/h-config.sh "$1"/h-run.sh "$1"/h-stats.sh
}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT

# 1) HiveOS — miner name = keryx-miner-supr-<line>
HVN="${NAME}-${LABEL}"
H="$S/hv/$HVN"; mkdir -p "$H"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$H/"
hrename "$H" "$HVN"
cp "$D/keryx-miner-supr" "$H/"; mklib "$H"; chmod +x "$H"/h-*.sh "$H"/keryx-miner-supr
tar -czf "$D/${HVN}-${VER}.tar.gz" -C "$S/hv" "$HVN"

# 2) SMOS — HiveOS layout, miner name = keryx-miner-supr-smos-<line>
SMN="${NAME}-smos-${LABEL}"
SM="$S/sm/$SMN"; mkdir -p "$SM"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$SM/"
hrename "$SM" "$SMN"
cp "$D/keryx-miner-supr" "$SM/"; mklib "$SM"; chmod +x "$SM"/h-*.sh "$SM"/keryx-miner-supr
tar -czf "$D/${SMN}-${VER}.tar.gz" -C "$S/sm" "$SMN"

# 3) mmpOS — folder = <name>-<line>-mmpos_<ver>, EXTERNAL_NAME set to the line
MMN="${NAME}-${LABEL}"
MM="$S/mm/${MMN}-mmpos_${VER}"; mkdir -p "$MM"
cp "$MPKG"/mmp-external.conf "$MPKG"/mmp-launch.sh "$MPKG"/mmp-stats.sh "$MM/"
sed -i -e "s|^EXTERNAL_NAME=.*|EXTERNAL_NAME=\"${MMN}\"|" -e "s|^EXTERNAL_VERSION=.*|EXTERNAL_VERSION=\"${VER}\"|" "$MM/mmp-external.conf"
cp "$D/keryx-miner-supr" "$MM/"; mklib "$MM"; chmod +x "$MM"/keryx-miner-supr "$MM"/*.sh
tar -czf "$D/${MMN}-mmpos_${VER}.tar.gz" -C "$S/mm" "${MMN}-mmpos_${VER}"

# 4) Generic Linux (dynamic binary + plugins + lib + run.sh)
LXN="${NAME}-${LABEL}"
LX="$S/lx/$LXN"; mkdir -p "$LX"
cp "$D/keryx-miner-supr-dynamic" "$LX/keryx-miner-supr"
cp "$D/libkeryxcuda.so" "$D/libkeryxopencl.so" "$LX/"; mklib "$LX"; chmod +x "$LX/keryx-miner-supr"
cat > "$LX/run.sh" <<'SH'
#!/usr/bin/env bash
cd "$(dirname "$(realpath "$0")")"
export LD_LIBRARY_PATH="$PWD:$PWD/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
exec ./keryx-miner-supr "$@"
SH
chmod +x "$LX/run.sh"
cat > "$LX/RUN.txt" <<TXT
keryx-miner-supr ${VER} (${LABEL}) — Linux x86_64
Bundled CUDA runtime in ./lib (this is the ${LABEL} build). Needs only the NVIDIA driver.
Run:  ./run.sh -a keryx:<addr>.<worker> -s stratum+tcp://krx.suprnova.cc:4401 --light --cuda-device 0
TXT
tar -czf "$D/${LXN}-${VER}-linux-x86_64.tar.gz" -C "$S/lx" "$LXN"

echo ">> ${LABEL} packages (in $D):"
( cd "$D" && ls -la *${LABEL}*.tar.gz && sha256sum *${LABEL}*.tar.gz > "SHA256SUMS-${LABEL}.txt" && cat "SHA256SUMS-${LABEL}.txt" )
