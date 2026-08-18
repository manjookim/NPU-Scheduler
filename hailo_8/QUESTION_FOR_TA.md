# Hailo-8 이식 중 발생한 기술적 문제 — 조교님 자문 요청

## 배경

기존에 Hailo-8L(Raspberry Pi 5, 보드 rpi1)에서 진행하던 NPU 스케줄러(priority/threshold/timeout) 실험을, 같은 실험을 다른 칩인 **Hailo-8**(역시 Raspberry Pi 5, 보드 rpi4)에서도 재현해보려고 환경을 이식하는 작업을 진행했습니다. HEF 파일 재확보, 소스 코드 경로 수정, 빌드까지는 문제없이 끝났는데, 실제 추론을 실행하는 단계에서 막혀서 자문을 구하고 싶습니다.

## 진행한 것 (문제없이 완료)

1. Hailo-8 칩 인식 확인 (`hailortcli fw-control identify` → `Device Architecture: HAILO8`, Firmware 4.23.0)
2. HailoRT 개발 헤더/빌드 도구 확인
3. Hailo-8용 HEF 3종 재확보 — 기존 8L용 `_h8l.hef`는 아키텍처가 달라 그대로 못 씀. Hailo Model Zoo **v2.14.0 / hailo8** 타겟으로 통일해서 재획득 (`yolov8s.hef`, `yolov8s_seg.hef`, `yolov8s_pose.hef`). `hailortcli parse-hef`로 3개 다 `Architecture: HAILO8` 확인
4. 데이터셋(`sampled_val2017`, 673장) 전송
5. 기존 8L용 `infer_scheduler.cpp`를 복사해 `infer_scheduler_hailo8.cpp` 생성, HEF 경로/`IMG_DIR`만 rpi4 환경에 맞게 수정, 나머지 코드(스케줄러 API 호출부)는 그대로. 빌드 성공

## 막힌 문제

3개 모델(Det/Seg/Pose) 동시 실행은 물론, **모델 1개(Detection)만 단독으로 실행해도** 첫 프레임 하나조차 주고받지 못하고 전부 `HAILO_TIMEOUT`이 발생합니다. 이후 디바이스가 완전히 응답 불가(`device inaccessible`) 상태가 되어 재부팅해야만 복구됩니다 (드라이버 모듈만 재로드해서는 복구 안 된 적도 있음).

### 원인 조사

`dmesg`를 확인하니 매번 동일한 커널 크래시 트레이스가 찍혀 있었습니다.

```
find_vma+0x6c/0x80
hailo_vdma_buffer_map+0x8c/0x5e8 [hailo_pci]
hailo_vdma_buffer_map_ioctl+0xdc/0x340 [hailo_pci]
hailo_vdma_ioctl+0xcc/0x260 [hailo_pci]
hailo_pcie_fops_unlockedioctl+0x17c/0x610 [hailo_pci]
...
hailo 0001:01:00.0: Unable to change power state from D0 to D3hot, device inaccessible
```

Hailo 공식 커뮤니티에서 동일한 증상이 이미 보고되어 있었습니다: `hailo_pci` 드라이버(v4.20.0~4.23.0)가 `hailo_vdma_buffer_map()` 내부에서 `find_vma()`를 `mmap_read_lock()`/`mmap_read_unlock()` 없이 호출하는 락킹 버그이고, Raspberry Pi 5의 최신 커널(6.12.x)에서 주로 재현된다고 합니다. 저희 rpi4 보드가 정확히 이 조합입니다.

- [hailo_pci VDMA buffer mapping 버그 리포트](https://community.hailo.ai/t/the-stack-trace-points-to-an-issue-with-vdma-buffer-mapping-in-the-hailo-pci-driver-4-20-1/15205)
- [find_vma() called before mmap_read_lock() 버그 리포트 (hailo_pci-4.23.0)](https://community.hailo.ai/t/bug-report-find-vma-called-before-mmap-read-lock-issued-in-usr-src-hailo-pci-4-23-0-linux-vdma-memory-c/19001)
- [hailo_pci 4.20.0 DKMS driver, RPi5 kernel 6.12.x에서 find_vma 경고 다수 발생](https://community.hailo.ai/t/hailo-pci-4-20-0-dkms-driver-triggers-120-find-vma-kernel-warns-during-model-configure-on-rpi-5-kernel-6-12-x/19079)

### 배제한 가설

- **HEF 파일 문제 아님**: `parse-hef`로 아키텍처 확인 완료, `Hef::create`/`configure`/`set_scheduler_*` 전부 성공한 뒤 실제 데이터 전송 단계에서만 실패.
- **저희 코드 문제 아님**: Hailo 공식 예제(`hailort` 저장소 `hailo8` 브랜치의 `switch_network_groups_example.cpp`)와 구조가 1:1로 일치함을 직접 대조 확인. 이 브랜치는 Hailo-8/8L/8R 공용이라 칩별로 다른 코드 자체가 없음.
- **3모델 동시 스케줄링 문제 아님**: 모델 1개만 돌려도 동일하게 크래시.
- **HailoRT 5.x 업그레이드는 해결책 아님**: Hailo-8/8L 라인은 애초에 5.x를 지원하지 않음(5.x는 Hailo-10/15 전용). `hailort-drivers` 저장소 확인 결과 4.x 라인의 최신(최후) 버전이 정확히 지금 설치된 4.23.0이라, 드라이버 업그레이드로 고칠 방법이 없음.

### 8L 보드와의 환경 비교

| | rpi4 (Hailo-8, 크래시 발생) | rpi1 (Hailo-8L, 정상 동작) |
|---|---|---|
| OS | Debian 13 (trixie) | Ubuntu 24.04.4 LTS |
| 커널 | 6.12.47+rpt-rpi-2712 | 6.8.0-1056-raspi |
| HailoRT/드라이버 | 4.23.0 | 4.23.0 (동일) |

드라이버/HailoRT 버전은 동일하고 **커널 버전만 다릅니다.** 버그가 보고된 6.12.x 계열과 8L 보드의 6.8.x 계열 차이가 안정성 차이의 핵심으로 추정하고 있습니다.

## 여쭤보고 싶은 것

1. rpi4 보드를 8L 보드(rpi1)와 동일한 Ubuntu 24.04 LTS 환경으로 재설치하는 게 맞을지, 아니면 다른 방법(커널만 다운그레이드 / 드라이버 소스 직접 패치)을 권장하시는지 궁금합니다.
2. 랩/학과 차원에서 Hailo-8 보드에 이미 검증된 특정 OS·커널 이미지가 있다면 알고 싶습니다.
3. 위 방법들이 다 여의치 않을 경우, "크래시 나면 재부팅 후 재시도"를 자동화한 워크어라운드로 진행해도 괜찮을지 (원래 8L 실험처럼 대량 반복 실행에 이 방식이 적합할지) 의견 부탁드립니다.

## 환경 정보

- 보드: Raspberry Pi 5 Model B, 호스트명 rpi4
- 칩: Hailo-8 (PCIe, M.2 HAT)
- OS: Debian GNU/Linux 13 (trixie), 커널 6.12.47+rpt-rpi-2712
- HailoRT / hailort-pcie-driver: 4.23.0
- 참고: 8L 보드(rpi1)는 Ubuntu 24.04.4 LTS, 커널 6.8.0-1056-raspi, HailoRT 4.23.0 (동일 버전)

## [2026-08-07 추가] 재발 — 이번엔 디바이스가 완전히 죽진 않음

`infer_scheduler_hailo8_v5det`(Detection=YOLOv5 nms_core HEF, 3모델 동시 실행) 실험 중
`run_id=1`은 정상 완료됐는데, `run_id=2` 시작 시점에 동일한 트레이스로 재발함
(`find_vma+0x6c/0x80` → `hailo_vdma_buffer_map` → `hailo_vdma_buffer_map_ioctl` →
`hailo_pcie_fops_unlockedioctl`, `WARNING: ... at include/linux/rwsem.h:80 find_vma+0x6c/0x80`,
`Tainted: G W O`). 모델 종류(YOLOv5 nms_core 포함)나 개수와 무관하게 터진다는 기존 판단과
일치함.

**차이점**: 이번엔 크래시 직후 `hailortcli fw-control identify`가 정상 응답함
(Firmware 4.23.0, Architecture HAILO8 그대로 확인) — 재부팅 없이 디바이스가 살아있는
상태로 유지됨. 예전 기록("완전히 응답 불가 상태가 되어 재부팅해야만 복구")과 다르게
이번엔 WARN만 찍히고 하드 크래시까지는 안 간 케이스. 재시도 시 프로세스 재실행만으로
넘어갈 수 있었음(재부팅 불필요). 다만 07-26 배치스윕 실험 등은 이 사이에 문제없이
완료된 이력이 있어, 이 버그가 상시 재현되는 게 아니라 간헐적으로(비결정적으로) 터지는
것으로 보임 — 자동화 스크립트에 "실패 시 자동 재시도" 로직을 넣는 게 실용적일지도
질문 3번과 함께 여쭤보고 싶음.
