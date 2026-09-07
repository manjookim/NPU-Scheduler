#!/bin/bash
# build.sh — infer_scheduler_h10 빌드 (npu-rpi5 / Raspberry Pi 5 + Hailo-10H)
# 사용법: chmod +x build.sh && ./build.sh
set -eu
cd "$(dirname "$0")"

SRC=infer_scheduler_h10.cpp
OUT=infer_scheduler_h10

echo "== 의존성 확인 =="
pkg-config --exists opencv4 || { echo "opencv4 pkg-config 없음. libopencv-dev 설치 필요"; exit 1; }
if ldconfig -p 2>/dev/null | grep -q libhailort; then
    echo "  libhailort: $(ldconfig -p | grep -m1 libhailort | sed 's/.*=> //')"
elif ls /usr/lib/libhailort.so* /usr/lib/*/libhailort.so* /usr/local/lib/libhailort.so* >/dev/null 2>&1; then
    echo "  libhailort: $(ls /usr/lib/libhailort.so* /usr/lib/*/libhailort.so* /usr/local/lib/libhailort.so* 2>/dev/null | head -1)"
else
    echo "  [경고] libhailort를 못 찾음 — 링크 실패하면 -L <경로>를 추가할 것"
fi
ls /usr/include/hailo/hailort.hpp >/dev/null 2>&1 \
    && echo "  헤더: /usr/include/hailo/hailort.hpp" \
    || echo "  [경고] /usr/include/hailo/hailort.hpp 없음 — -I <경로> 필요"

echo "== 빌드 =="
g++ -O2 -std=c++17 "$SRC" -o "$OUT" -I. \
    $(pkg-config --cflags --libs opencv4) \
    -lhailort -lpthread

echo "== 완료: ./$OUT =="
echo
echo "빠른 확인 (det 단독 60장, CSV 저장 없이):"
echo "  ./$OUT 1 --models det --num_images 60 --api vstreams"
echo "  ./$OUT 1 --models det --num_images 60 --api infer"                 # in-flight 자동(8L 매칭)
echo "  ./$OUT 1 --models det,seg,pose --api infer --inflight 11,9,9"      # 모델별 직접 지정
