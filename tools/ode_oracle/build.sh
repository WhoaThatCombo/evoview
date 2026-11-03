#!/bin/sh
# Builds the reference ODE 0.15.2 (double precision, OPCODE, no FMA
# contraction so it rounds like the Rust port) and the oracle against it.
# usage: build.sh <ode-0.15.2 source dir> [out dir]
set -e
SRC=$(cd "$1" && pwd)
OUT=${2:-$(pwd)/out}
mkdir -p "$OUT/obj"
cd "$SRC"
[ -f include/ode/precision.h ] || sed 's/@ODE_PRECISION@/dDOUBLE/' include/ode/precision.h.in > include/ode/precision.h
[ -f include/ode/version.h ] || sed 's/@ODE_VERSION@/0.15.2/' include/ode/version.h.in > include/ode/version.h
if [ ! -f ode/src/config.h ]; then
  sed 's#/\* \#define dOU_ENABLED 1 \*/#\#define dOU_ENABLED 1#; s#/\* \#define dATOMICS_ENABLED 1 \*/#\#define dATOMICS_ENABLED 1#; s#/\* \#define dBUILTIN_THREADING_IMPL_ENABLED 1 \*/#\#define dBUILTIN_THREADING_IMPL_ENABLED 1#' build/config-default.h > ode/src/config.h
fi
FLAGS="-O2 -msse2 -ffp-contract=off -DNDEBUG -DdNODEBUG -DdIDEDOUBLE -DODE_LIB -D_USE_MATH_DEFINES -D_OU_NAMESPACE=odeou -D_OU_FEATURE_SET=_OU_FEATURE_SET_ATOMICS -Iinclude -Iode/src -Iode/src/joints -IOPCODE -IOPCODE/Ice -Iou/include -w"
if [ ! -f "$OUT/libode.a" ]; then
  for f in $(ls ode/src/*.cpp ode/src/joints/*.cpp ode/src/*.c OPCODE/*.cpp OPCODE/Ice/*.cpp ou/src/ou/*.cpp 2>/dev/null | grep -vE 'gimpact|libccd|collision_trimesh_disabled|threading_pool_posix|export-dif'); do
    o="$OUT/obj/$(echo $f | tr '/' '_').o"
    [ -f "$o" ] || g++ -c $FLAGS "$f" -o "$o"
  done
  ar rcs "$OUT/libode.a" "$OUT"/obj/*.o
fi
cd - >/dev/null
g++ -O2 -msse2 -ffp-contract=off -DdIDEDOUBLE -I"$SRC/include" "$(dirname "$0")/oracle.cpp" "$OUT/libode.a" -o "$OUT/ode_oracle.exe" -lpthread
echo "$OUT/ode_oracle.exe"
