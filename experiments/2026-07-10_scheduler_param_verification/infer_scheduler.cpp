/**
 * infer_scheduler.cpp
 * ------------------------------------------------------------------------
 * Hailo-8L (Raspberry Pi 5) 위에서 Detection / Segmentation / Pose 세 모델을
 * HailoRT Model Scheduler(ROUND_ROBIN)로 "동시에" 추론하면서, 모델별로
 *   priority / threshold / timeout / batch_size
 * 를 독립적으로 설정해 벤치마킹하기 위한 C++ 프로그램.
 *
 * 입력 데이터: COCO val2017 샘플 이미지(IMG_DIR, 기본 최대 600장) — 더미 버퍼가 아닌
 *            실제 이미지를 읽어 letterbox 전처리 후 추론에 사용한다.
 *
 * 스케줄러 파라미터 적용 근거 (공식 문서/예제, github.com/hailo-ai/hailort, hailo8 브랜치):
 *   - hailort/libhailort/include/hailo/network_group.hpp
 *       set_scheduler_timeout(const std::chrono::milliseconds&, network_name="")
 *         -> 기본값 0ms. "적어도 한 번의 요청이 들어온 뒤" 이 시간이 지나면
 *            threshold 미달이어도 강제로 실행 자격을 얻는다.
 *       set_scheduler_threshold(uint32_t, network_name="")
 *         -> 기본값 1. 큐에 threshold개 요청이 쌓여야 실행 자격을 얻는다(단, timeout이
 *            먼저 지나면 그 전에도 자격을 얻음 — 위 timeout 설명 참고).
 *       set_scheduler_priority(uint8_t, network_name="")
 *         -> 기본값 HAILO_SCHEDULER_PRIORITY_NORMAL. 값이 클수록 우선.
 *       (주의: 세 함수 모두 "network_name 지정 시 특정 네트워크만 설정"은 아직 미지원 —
 *        네트워크그룹 전체 단위로만 적용됨. 본 코드는 모델당 네트워크그룹을 하나씩
 *        따로 configure하므로 문제 없음.)
 *   - hailort/libhailort/examples/cpp/switch_network_groups_example/switch_network_groups_example.cpp
 *       VDevice::create -> create_configure_params(hef) -> batch_size 설정 -> configure(hef)
 *       -> set_scheduler_timeout/threshold/priority -> VStreamsBuilder::create_vstreams(*ng, {}, FORMAT_TYPE)
 *     본 파일의 구조/시그니처는 위 공식 예제를 그대로 따른다.
 *
 * 참고(프로젝트 문서): PROJECT_HANDOFF.md, README.md, docs/setup.md, memory/findings.md
 *   - threshold 효과를 실제로 관측하려면 timeout > 0 이어야 하고(§7),
 *     입력을 한꺼번에 다 밀어넣지 말고 NPU 처리량 근처 속도로 흘려보내야
 *     큐가 상시 포화되지 않아 threshold/timeout이 의미를 가진다.
 *     -> 아래 INPUT_FPS 로 입력 속도를 제한할 수 있다 (0 = 제한 없음/최대속도).
 *
 * 빌드 (RPi):
 *   g++ infer_scheduler.cpp -o infer_scheduler -lhailort $(pkg-config --cflags --libs opencv4) -lpthread -std=c++17
 *
 * 실행:
 *   ./infer_scheduler [run_id]
 *
 * 현재는 CSV 저장 없이, 콘솔의 [적용확인] 로그 + HRTT 트레이스만으로 파라미터
 * 적용 여부를 확인하는 것이 목적이다. 아래 환경변수를 설정하고 실행하면 HRTT가
 * 생성되고, PC/WSL에서 `hailo runtime-profiler <파일>.hrtt`로 변환한 HTML에서
 * `core_op_set_value` 이벤트로 실제 적용된 threshold/timeout/priority 값을 확인할 수 있다
 * (PROJECT_HANDOFF.md §6 참고. setter가 HAILO_SUCCESS를 반환해도 실제 반영은
 * HRTT로 재확인하는 것이 정확함).
 *
 * HRTT 트레이스:
 *   export HAILO_TRACE=scheduler
 *   export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30
 *   export HAILO_TRACE_PATH=/path/to/traces
 *   export HAILO_MONITOR=1
 * ------------------------------------------------------------------------
 */

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <algorithm>
#include <unistd.h>

using namespace hailort;

// ========================= 파라미터 설정 (모델별로 다르게) =========================
// [실기 확인된 제약, 공식 문서에 명시 안 됨] threshold는 반드시 그 모델의 batch_size
// 이하여야 한다. 초과 시 set_scheduler_threshold가 HAILO_INVALID_ARGUMENT로 실패하고
// (HailoRT 로그: "Threshold must be equal or lower than the maximum batch size!"),
// 해당 모델은 threshold가 기본값(1)으로 남는다 — [적용확인] 로그의 [실패] 표시로 알 수 있음.
// 즉 아래 THRESHOLD_* <= BATCH_* 를 항상 지킬 것.

// 모델별 batch_size (네트워크그룹 입출력 큐 크기 — 공식 문서: 클수록 pre/post-process가
// 하드웨어 추론과 병렬화되어 다른 스케줄러 파라미터가 더 잘 작동함)
#define BATCH_DET       4
#define BATCH_SEG       8
#define BATCH_POSE      2

// threshold: 네트워크그룹이 스케줄될 "자격"을 얻기 위한 최소 누적 요청 수 (기본값 1)
// 반드시 threshold <= 같은 줄의 batch_size (위 제약 참고)
// [검증 완료] threshold(99) > batch(8)로 깨서 테스트한 결과, core_op_set_value 트레이스에
// 거부된 값은 실제로 안 남는 것을 확인함(verify_params.py로 (미적용!) 확인). 정상값으로 복구.
#define THRESHOLD_DET   4
#define THRESHOLD_SEG   6
#define THRESHOLD_POSE  2

// timeout(ms): threshold 미달이어도 최소 1프레임 + 이 시간이 지나면 강제로 실행 자격 부여 (기본값 0)
#define TIMEOUT_DET_MS   200
#define TIMEOUT_SEG_MS   500
#define TIMEOUT_POSE_MS  100

// priority: 0~31, 클수록 우선 (기본값 16=NORMAL). 스케줄 가능한 여러 모델 중 이 값이
// 가장 큰 모델부터 확인한다. 동일하면 Round-Robin.
#define PRIORITY_DET    15
#define PRIORITY_SEG    15
#define PRIORITY_POSE   15

// 어떤 모델을 이번 실행에 포함할지 (0/1)
#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

// 입력 속도 제한 (모델당 초당 프레임 수). 0 = 제한 없음(최대 속도로 큐를 채움 -> 큐가
// 항상 포화 상태가 되어 threshold/timeout 효과가 거의 관측되지 않음, findings.md 참고).
// threshold/timeout 효과를 보고 싶다면 0보다 큰 값(예: NPU 처리량 근처)으로 설정할 것.
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
#define NUM_IMAGES      600
// =====================================================================================

// HEF 경로 (Raspberry Pi 5, hailo-rpi5-examples 리소스)
#define DET_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"

// val2017 데이터셋 경로 (docs/setup.md 기준 기본 경로).
// 라즈베리파이에 실제로 저장된 경로가 다르면 이 값만 수정 후 재컴파일할 것
// (예: sampled_val2017을 쓰는 경우 "/home/rpi1/datasets/sampled_val2017/").
#define IMG_DIR  "/home/rpi1/datasets/coco/val2017/"

std::mutex print_mutex;

// ========================= 시스템 모니터링 (CPU/MEM/Context Switch) =========================

struct CpuStats {
    long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
};

CpuStats read_cpu_stats() {
    CpuStats s;
    std::ifstream f("/proc/stat");
    std::string line;
    std: