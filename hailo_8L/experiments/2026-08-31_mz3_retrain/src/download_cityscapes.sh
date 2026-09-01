#!/bin/bash
# Cityscapes 다운로드 (계정 필요: https://www.cityscapes-dataset.com/register/)
# 사용: CS_USER=아이디 CS_PASS=비번 bash download_cityscapes.sh /경로/Cityscapes
set -eu
DEST="${1:-./Cityscapes}"
: "${CS_USER:?CS_USER 환경변수에 아이디를 넣으세요}"
: "${CS_PASS:?CS_PASS 환경변수에 비밀번호를 넣으세요}"
mkdir -p "$DEST" && cd "$DEST"

wget --keep-session-cookies --save-cookies=cookies.txt -O /dev/null \
  --post-data "username=${CS_USER}&password=${CS_PASS}&submit=Login" \
  https://www.cityscapes-dataset.com/login/

echo "[1/2] gtFine_trainvaltest.zip (약 253MB)"
wget -c --load-cookies cookies.txt --content-disposition \
  "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
echo "[2/2] leftImg8bit_trainvaltest.zip (약 11GB)"
wget -c --load-cookies cookies.txt --content-disposition \
  "https://www.cityscapes-dataset.com/file-handling/?packageID=3"

unzip -q -o gtFine_trainvaltest.zip
unzip -q -o leftImg8bit_trainvaltest.zip
rm -f cookies.txt

echo "완료. 구조 확인:"
ls -d gtFine/train gtFine/val leftImg8bit/train leftImg8bit/val
echo "train 이미지 수: $(find leftImg8bit/train -name '*_leftImg8bit.png' | wc -l)  (정상=2975)"
echo "val   이미지 수: $(find leftImg8bit/val   -name '*_leftImg8bit.png' | wc -l)  (정상=500)"
