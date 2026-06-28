#!/usr/bin/env bash
# Package one build "line" (legacy or modern) into all distribution formats, using
# the binaries + CUDA runtime libs already produced in hiveos/<DISTDIR>/ by
# build-offline.sh. The bundled libs ARE the line's CUDA version (legacy=12.2/
# floor535, modern=12.9/floor575), so each package carries its own driver floor.
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
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT

# 1) HiveOS (static binary + lib + h-*.sh)
H="$S/hv/$NAME"; mkdir -p "$H"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$H/"
cp "$D/keryx-miner-supr" "$H/"; mklib "$H"; chmod +x "$H"/h-*.sh "$H"/keryx-miner-supr
tar -czf "$D/${NAME}-${VER}-${LABEL}.tar.gz" -C "$S/hv" "$NAME"

# 2) SMOS (HiveOS layout, smos-named)
SM="$S/sm/${NAME}-smos_${VER}-${LABEL}"; mkdir -p "$SM"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$SM/"
cp "$D/keryx-miner-supr" "$SM/"; mklib "$SM"; chmod +x "$SM"/h-*.sh "$SM"/keryx-miner-supr
tar -czf "$D/${NAME}-smos_${VER}-${LABEL}.tar.gz" -C "$S/sm" "${NAME}-smos_${VER}-${LABEL}"

# 3) mmpOS (static binary + lib + mmp-*.sh)
MM="$S/mm/${NAME}-mmpos_${VER}-${LABEL}"; mkdir -p "$MM"
cp "$MPKG"/mmp-external.conf "$MPKG"/mmp-launch.sh "$MPKG"/mmp-stats.sh "$MM/"
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"${VER}-${LABEL}\"/" "$MM/mmp-external.conf"
cp "$D/keryx-miner-supr" "$MM/"; mklib "$MM"; chmod +x "$MM"/keryx-miner-supr "$MM"/*.sh
tar -czf "$D/${NAME}-mmpos_${VER}-${LABEL}.tar.gz" -C "$S/mm" "${NAME}-mmpos_${VER}-${LABEL}"

# 4) Generic Linux (dynamic binary + plugins + lib + run.sh)
LX="$S/lx/$NAME"; mkdir -p "$LX"
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
tar -czf "$D/${NAME}-${VER}-${LABEL}-linux-x86_64.tar.gz" -C "$S/lx" "$NAME"

echo ">> ${LABEL} packages (in $D):"
( cd "$D" && ls -la *-${LABEL}*.tar.gz && sha256sum *-${LABEL}*.tar.gz > "SHA256SUMS-${LABEL}.txt" && cat "SHA256SUMS-${LABEL}.txt" )
