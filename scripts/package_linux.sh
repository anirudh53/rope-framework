#!/usr/bin/env bash
# package_linux.sh -- build portable Linux release archives (cpu, cuda12) of rope-framework.
#
#   scripts/package_linux.sh <version> <out_dir> [model_dir ...]
#
# Needs a toolchain environment with cmake >= 4.2, a C++20 compiler (GCC >= 12), ninja,
# cpp-httplib, openssl, zlib, zstd and patchelf (e.g. conda-forge: cmake ninja gxx_linux-64=13
# cpp-httplib openssl zlib zstd patchelf), activated, with $CONDA_PREFIX set.
#
# Portability choices:
#   * generic x86-64 code (-O3, no -march=native)
#   * librope.so (C API, no ONNX Runtime dependency) links libstdc++/libgcc statically, so it
#     loads into any host process (Python, .NET) whatever libstdc++ that process already has
#   * bin/rope uses the bundled libstdc++/libgcc_s/libgomp/openssl/zlib/zstd from lib/
#   * RPATHs are archive-relative only ($ORIGIN/../lib for bin/, $ORIGIN for lib/)
# Each model_dir given is copied to data/models/; the first becomes the default in config/rope.conf.
set -euo pipefail

VERSION=${1:?version}; OUT=${2:?out_dir}; shift 2
MODELS=("$@")
SRC=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${CONDA_PREFIX:?activate the toolchain environment first}
export CC=${CC:-x86_64-conda-linux-gnu-gcc} CXX=${CXX:-x86_64-conda-linux-gnu-g++}
mkdir -p "$OUT"

for HW in cpu cuda12; do
    B="$SRC/build-rel-$HW"
    cmake -S "$SRC" -B "$B" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG" -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O3 -DNDEBUG" \
        -DROPE_DOWNLOAD_DEPS=ON -DROPE_HARDWARE=$HW -DROPE_USE_LIBTORCH=OFF \
        -DROPE_BUILD_TESTS=$([ $HW = cpu ] && echo ON || echo OFF) \
        -DROPE_RELEASE_VERSION="$VERSION" -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DCMAKE_CXX_STANDARD_LIBRARIES="-lanl" \
        -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
    cmake --build "$B" --parallel
    if [ $HW = cpu ]; then ctest --test-dir "$B" --output-on-failure; fi

    S="$OUT/rope_framework-$VERSION-linux-x86_64-$HW"
    rm -rf "$S"
    cmake --install "$B" --prefix "$S"
    find "$S" -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
    for l in libstdc++.so.6 libgcc_s.so.1 libgomp.so.1 libssl.so.3 libcrypto.so.3 libz.so.1 libzstd.so.1; do
        cp -L "$PREFIX/lib/$l" "$S/lib/"
    done
    patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "$S/bin/rope"
    patchelf --force-rpath --set-rpath '$ORIGIN' "$S/lib/librope.so"

    if [ ${#MODELS[@]} -gt 0 ]; then
        mkdir -p "$S/data/models"
        for m in "${MODELS[@]}"; do cp -a "$m" "$S/data/models/"; done
        sed -i "s#^exported_dir = .*#exported_dir = ../data/models/$(basename "${MODELS[0]}")#" "$S/config/rope.conf"
    fi

    # self-check: nothing may resolve outside the archive or the base system
    if env -i PATH=/usr/bin:/bin ldd "$S/bin/rope" "$S/lib/librope.so" | grep -E "not found|$PREFIX"; then
        echo "package_linux.sh: $S has unresolved or non-portable dependencies" >&2; exit 1
    fi
    tar -C "$OUT" -czf "$S.tar.gz" "$(basename "$S")"
    echo "packaged $S.tar.gz"
done
