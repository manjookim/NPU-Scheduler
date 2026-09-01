# NPU Scheduler — 프로젝트 지침

## ⚠️ 보드 / 계정 (가장 먼저 확인할 것)

**Hailo-8L 보드의 계정 ID는 `npu-rpi1` 입니다.** (구: `rpi1` — 2026-08-20 계정 이관)

```bash
ssh npu-rpi1@155.230.16.157 -p 40021
```

- 홈 디렉터리도 `/home/npu-rpi1/` 로 바뀌었습니다. **하드코딩된 `/home/rpi1/` 경로는 전부 깨집니다.**
- 옛 문서(`memory/rpi_connection.md`, `PROJECT_SUMMARY.md`, 일부 `.cpp`/`.sh`)에 `rpi1@` 또는
  `/home/rpi1/` 이 남아 있을 수 있습니다. 보이면 `npu-rpi1` 로 고쳐 쓰세요.

### 보드 목록

| 별칭 | 보드 | NPU | HailoRT | 비고 |
|---|---|---|---|---|
| **npu-rpi1** | Raspberry Pi 5 | **Hailo-8L** | 4.23.0 | 주 실험 보드. HRTT 트레이싱 가능 |
| npu-rpi2 | Raspberry Pi 5 | Hailo-8L | — | **조교님 기기** (대조용, 우리가 쓰는 게 아님) |
| npu-rpi5 | Raspberry Pi 5 | Hailo-10H | 5.3.0 | HRTT **불가** (추론 스택이 디바이스 내부) |
| rpi4 | Raspberry Pi 5 | Hailo-8 | — | 2026-08-19 이후 사용 중단 |

PCIe 는 세 보드 모두 **Gen3 x1**. Hailo 공식 Model Zoo FPS 는 Gen3 x2 기준이라 대역폭이 약 절반입니다.

---

## 현재 작업 (2026-09-01)

`workload2.xlsx` SMART TRAFFIC 시나리오 2 의 3모델을 Hailo-8L에서 Default 조건으로 벤치마크.

| 모델 | 재학습 | HEF | 측정 |
|---|---|---|---|
| `ssd_mobilenet_v1` (Det, COCO) | 불필요 | MZ 공식 | ✅ 42런 |
| `deeplab_v3_mobilenet_v2_wo_dilation` (Seg) | **필요** (PASCAL VOC → Cityscapes) | MZ 공식 | ✅ 42런 (사전학습 기준) |
| `mobilenet_v2_1.0` (Cls) | ✅ GTSRB 43클래스 | 자체 컴파일 | ⬜ 재학습판 미측정 |

상세 가이드: **`memory/2026-08-31_mz3_retrain_baseline_guide.md`**

---

## 작업 규칙

- **노션 정리 요청 = 채팅에 마크다운으로.** Notion MCP 로 페이지를 직접 만들거나 수정하지 말 것
  (`memory/notion_workflow.md` 참조).
- **DFC 는 x86 전용** — RPi 에서 컴파일 불가. PC/WSL 에서 컴파일하고 HEF 만 scp.
- **Model Zoo 버전이 보드별로 다름**: Hailo-8/8L 컴파일은 **v2.19.0 이하 + DFC 3.3x**,
  Hailo-10H 는 v5.x + DFC 5.x. master(v5.x) yaml 에는 `hailo8l` 이 빠져 있습니다.
- **scp 경로 끝에 `/` 나 `\` 를 붙이지 말 것** — bash 에서 EOF 에러가 납니다.
- 실험 재시작 전 결과 CSV·traces 정리 (데이터 오염 방지).
- 이 저장소에서 `git status` 등 인덱스를 쓰는 명령은 lock 파일을 남길 수 있습니다
  (`.git/index.lock`). 남으면 `Remove-Item .git\index.lock -Force`.

## 문서 지도

| 파일 | 내용 |
|---|---|
| `memory/MEMORY.md` | 메모리 인덱스 |
| `memory/2026-08-31_mz3_retrain_baseline_guide.md` | 3모델 재학습·컴파일·베이스라인 설계·트러블슈팅 |
| `PROJECT_SUMMARY.md` | 프로젝트 전체 이력 (2026-08-19 기준, 계정명 일부 구버전) |
| `memory/findings.md` | starvation, 스케줄러 API 특성 등 핵심 발견 |
| `hailo_10h/.../HRTT_ON_HAILO10H.md` | Hailo-10H 측정 한계와 대체 수단 |
