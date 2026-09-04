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

[[ -s "$D/keryx-miner-supr" ]] || { echo "ERROR: $D/keryx-miner-supr (static) missing"; exit 1; }
[[ -s "$D/keryx-miner-supr-dynamic" ]] || { echo "ERROR: $D/keryx-miner-supr-dynamic missing"; exit 1; }
[[ -s "$D/libkeryxcuda.so" ]] || { echo "ERROR: $D/libkeryxcuda.so missing"; exit 1; }
LIBS=(libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10)
for runtime in "${LIBS[@]}"; do
  [[ -s "$D/lib/$runtime" ]] || { echo "ERROR: required CUDA runtime missing: $D/lib/$runtime"; exit 1; }
done

mklib(){ mkdir -p "$1/lib"; for l in "${LIBS[@]}"; do cp -L "$D/lib/$l" "$1/lib/"; done; }
# Both in-process llama.cpp engines are release-critical: libkeryx-llama.so is the normal
# CUDA engine, while libkeryx-llama-noavx.so keeps that same GPU route usable on rig CPUs that
# cannot safely load the AVX2 build. H6 has no non-llama GPU fallback, and CPU inference is a
# deprecated explicit emergency path, so never publish a line package without both engines.
for engine in libkeryx-llama.so libkeryx-llama-noavx.so; do
  [[ -s "$D/$engine" ]] || {
    echo "ERROR: required NVIDIA GPU inference engine missing: $D/$engine" >&2
    echo "       Re-run hiveos/build-offline.sh for the $LABEL line before packaging." >&2
    exit 1
  }
done

bundle_llama(){
  cp "$D/libkeryx-llama.so" "$D/libkeryx-llama-noavx.so" "$1/"
  chmod +x "$1/libkeryx-llama.so" "$1/libkeryx-llama-noavx.so"
  # llama-server is only an optional subprocess fallback for the in-process engines.
  if [ -f "$D/llama-server" ]; then cp "$D/llama-server" "$1/"; chmod +x "$1/llama-server"; fi
}
# Rewrite the HiveOS miner-dir name in the hardcoded h-* paths + CUSTOM_NAME (the
# binary stays ./keryx-miner-supr; only the /hive + /var/log dir component changes).
hrename(){ # $1=dir of h-* files  $2=new miner name
  sed -i -e "s|^CUSTOM_NAME=.*|CUSTOM_NAME=$2|" \
         -e "s|/hive/miners/custom/keryx-miner-supr/|/hive/miners/custom/$2/|g" \
         -e "s|/var/log/miner/keryx-miner-supr/|/var/log/miner/$2/|g" \
         "$1"/h-manifest.conf "$1"/h-config.sh "$1"/h-run.sh "$1"/h-stats.sh
}
S=$(mktemp -d); trap 'rm -rf "$S"' EXIT

# mmpOS support files (mmp-external.conf, mmp-launch.sh + mmp-launcher.sh compat alias,
# mmp-stats.sh, mmp-release-notes.txt) go into EVERY flavor, not just the mmpos_ tarball:
# field reports show mmpOS users pointing the agent at the HiveOS/linux tarballs and getting
# "missing mmp-stats.sh / mmp-launcher.sh" — extra files are inert on HiveOS/SMOS/plain Linux.
bundle_mmp(){ # $1=dest dir  $2=miner name for EXTERNAL_NAME/stats NAME
  cp "$MPKG"/mmp-external.conf "$MPKG"/mmp-launch.sh "$MPKG"/mmp-launcher.sh "$MPKG"/mmp-stats.sh "$MPKG"/mmp-release-notes.txt "$1/"
  sed -i -e "s|^EXTERNAL_NAME=.*|EXTERNAL_NAME=\"$2\"|" -e "s|^EXTERNAL_VERSION=.*|EXTERNAL_VERSION=\"${VER}\"|" "$1/mmp-external.conf"
  # Keep the stats reporter's NAME in step with EXTERNAL_NAME (the line-suffixed name), so the
  # fallback log-parse path reports the same miner name the package is installed under.
  sed -i -e "s|^NAME=.*|NAME=\"$2\"|" "$1/mmp-stats.sh"
  chmod +x "$1"/mmp-launch.sh "$1"/mmp-launcher.sh "$1"/mmp-stats.sh
}

# 1) HiveOS — miner name = keryx-miner-supr-<line>
HVN="${NAME}-${LABEL}"
H="$S/hv/$HVN"; mkdir -p "$H"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$H/"
hrename "$H" "$HVN"
cp "$D/keryx-miner-supr" "$H/"; bundle_llama "$H"; bundle_mmp "$H" "$HVN"; mklib "$H"; chmod +x "$H"/h-*.sh "$H"/keryx-miner-supr
tar -czf "$D/${HVN}-${VER}.tar.gz" -C "$S/hv" "$HVN"

# 2) SMOS — HiveOS layout, miner name = keryx-miner-supr-smos-<line>
SMN="${NAME}-smos-${LABEL}"
SM="$S/sm/$SMN"; mkdir -p "$SM"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$SM/"
hrename "$SM" "$SMN"
cp "$D/keryx-miner-supr" "$SM/"; bundle_llama "$SM"; bundle_mmp "$SM" "$SMN"; mklib "$SM"; chmod +x "$SM"/h-*.sh "$SM"/keryx-miner-supr
tar -czf "$D/${SMN}-${VER}.tar.gz" -C "$S/sm" "$SMN"

# 3) mmpOS — folder = <name>-<line>-mmpos_<ver>, EXTERNAL_NAME set to the line
MMN="${NAME}-${LABEL}"
MM="$S/mm/${MMN}-mmpos_${VER}"; mkdir -p "$MM"
cp "$D/keryx-miner-supr" "$MM/"; bundle_llama "$MM"; bundle_mmp "$MM" "$MMN"; mklib "$MM"; chmod +x "$MM"/keryx-miner-supr
tar -czf "$D/${MMN}-mmpos_${VER}.tar.gz" -C "$S/mm" "${MMN}-mmpos_${VER}"

# 4) Generic NVIDIA Linux (dynamic pom-cuda binary + CUDA plugin + lib + run.sh).
# Never add libkeryxopencl.so here: a host has one compile-time PoM backend, and
# this line's host routes every post-fork worker through pom-cuda.
LXN="${NAME}-${LABEL}"
LX="$S/lx/$LXN"; mkdir -p "$LX"
cp "$D/keryx-miner-supr-dynamic" "$LX/keryx-miner-supr"
cp "$D/libkeryxcuda.so" "$LX/"; bundle_llama "$LX"; bundle_mmp "$LX" "$LXN"; mklib "$LX"; chmod +x "$LX/keryx-miner-supr"
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
NVIDIA-only: mixed-vendor rigs must run AMD cards in a separate pom-opencl process.
Run:  ./run.sh -a keryx:<addr>.<worker> -s stratum+tcp://krx.suprnova.cc:4401 --tier auto --cuda-device 0
TXT
tar -czf "$D/${LXN}-${VER}-linux-x86_64.tar.gz" -C "$S/lx" "$LXN"

echo ">> ${LABEL} packages (in $D):"
( cd "$D" && ls -la *${LABEL}*.tar.gz && sha256sum *${LABEL}*.tar.gz > "SHA256SUMS-${LABEL}.txt" && cat "SHA256SUMS-${LABEL}.txt" )
