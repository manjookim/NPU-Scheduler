#include <hailo/hailort.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <string>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <unistd.h>

// ========== 파라미터 설정 ==========
#define BATCH_SIZE      1
#define NUM_IMAGES      0     // 모델당 사용할 검증 이미지 수 (0 = 전체 사용, sampled_val2017 = 673장)
#define INPUT_FPS       0     // 입력 속도 제한(모델당 초당 프레임). 0 = 무제한(최대속도).
                              // >0이면 큐가 항상 꽉 차지 않아 threshold 효과가 관찰됨(실시간 스트리밍 모사)
#define THRESHOLD       1
#define THRESHOLD_EQ_PRIORITY 0  // 1: 각 모델 threshold = 해당 모델 priority (threshold 실험)
                                 // 0: 모든 모델 threshold = THRESHOLD 고정값 (기존 실험)
#define THRESHOLD_PER_MODEL 1    // 1: 모델별 독립 threshold(THRESHOLD_DET/SEG/POSE) 사용 (priority×threshold 조합 실험)
#define THRESHOLD_DET   1
#define THRESHOLD_SEG   1
#define THRESHOLD_POSE  1
#define TIMEOUT_MS      0
#define PRIORITY_DET    0
#define PRIORITY_SEG    0
#define PRIORITY_POSE   0

#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1
// ====================================

#define DET_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"
#define IMG_DIR  "/home/rpi1/datasets/sampled_val2017/"
// CSV_PATH는 argv[2]로 런타임에 전달받음 (기본값은 하드코딩 경로)
#define CSV_PATH_DEFAULT "/home/rpi1/hailo_cpp_test/results_all.csv"

#define DET_OUTPUT_COUNT  1
#define SEG_OUTPUT_COUNT  10
#define POSE_OUTPUT_COUNT 9

std::mutex print_mutex;

// ========== 시스템 모니터링 ==========

struct CpuStats {
    long user, nice, system, idle, iowait, irq, softirq;
};

CpuStats read_cpu_stats() {
    CpuStats s = {};
    std::ifstream f("/proc/stat");
    std::string line;
    std::getline(f, line);
    sscanf(line.c_str(), "cpu %ld %ld %ld %ld %ld %ld %ld",
           &s.user, &s.nice, &s.system, &s.idle, &s.iowait, &s.irq, &s.softirq);
    return s;
}

double calc_cpu_usage(CpuStats& s1, CpuStats& s2) {
    long idle1 = s1.idle + s1.iowait;
    long idle2 = s2.idle + s2.iowait;
    long total1 = s1.user + s1.nice + s1.system + s1.idle + s1.iowait + s1.irq + s1.softirq;
    long total2 = s2.user + s2.nice + s2.system + s2.idle + s2.iowait + s2.irq + s2.softirq;
    return 100.0 * (1.0 - (double)(idle2 - idle1) / (double)(total2 - total1));
}

double read_mem_usage() {
    std::ifstream f("/proc/meminfo");
    long total = 0, available = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("MemTotal:") != std::string::npos)
            sscanf(line.c_str(), "MemTotal: %ld kB", &total);
        if (line.find("MemAvailable:") != std::string::npos)
            sscanf(line.c_str(), "MemAvailable: %ld kB", &available);
    }
    return 100.0 * (1.0 - (double)available / (double)total);
}

// 프로세스별 context switch: /proc/[PID]/status에서 읽기
struct CtxSwitches {
    long voluntary;
    long nonvoluntary;
};

// 호출한 스레드 자신의 context switch 읽기.
// /proc/thread-self/status 는 현재 실행 중인 "스레드"의 상태를 가리킴 (Linux 3.17+).
// 주의: /proc/[PID]/status(메인 스레드)만 읽으면 추론이 도는 워커 스레드의 switch가 누락됨.
//       또 워커 스레드는 join 시점에 이미 종료돼 /proc/[PID]/task/* 에서도 사라지므로,
//       각 추론 스레드가 자기 자신의 값을 측정해 합산하는 방식이 정확함.
CtxSwitches read_thread_ctx_switches() {
    CtxSwitches cs = {0, 0};
    std::ifstream f("/proc/thread-self/status");
    std::string line;
    long v;
    while (std::getline(f, line)) {
        // sscanf는 문자열 맨 앞부터 매칭하므로, "nonvoluntary..." 줄에서
        // "voluntary..." 포맷은 자동으로 실패함 (부분문자열 오매칭 방지).
        // 반드시 nonvoluntary를 먼저 검사할 필요는 없으나, 명확성을 위해 분리.
        if (sscanf(line.c_str(), "nonvoluntary_ctxt_switches: %ld", &v) == 1)
            cs.nonvoluntary = v;
        else if (sscanf(line.c_str(), "voluntary_ctxt_switches: %ld", &v) == 1)
            cs.voluntary = v;
    }
    return cs;
}

// ========== 이미지 유틸 ==========

// Letterbox: 비율 유지 resize + gray(114) 패딩 → 640x640
// YOLOv8 학습 전처리와 동일 (RoundRobin.py letterbox 참고)
cv::Mat letterbox(const cv::Mat& img, int target_size = 640) {
    int orig_h = img.rows;
    int orig_w = img.cols;

    float scale = std::min((float)target_size / orig_h, (float)target_size / orig_w);
    int new_h = (int)(orig_h * scale);
    int new_w = (int)(orig_w * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // 패딩 계산 (중앙 배치)
    int pad_top    = (target_size - new_h) / 2;
    int pad_bottom = target_size - new_h - pad_top;
    int pad_left   = (target_size - new_w) / 2;
    int pad_right  = target_size - new_w - pad_left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

std::vector<std::string> get_image_files(const char* dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path);
    if (!dir) return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name.find(".jpg") != std::string::npos)
            files.push_back(std::string(dir_path) + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());  // 실험 재현성을 위해 정렬
    return files;
}

// ========== 비동기 추론 (producer/consumer) ==========
// writer 스레드가 입력 큐를 계속 채우고(큐가 가득 차면 write가 블로킹됨),
// reader 스레드가 출력을 뽑는다. 입력 큐가 '점진적으로' 쌓여야 scheduler의
// threshold(큐에 N개 쌓이면 활성화)/timeout이 실제로 의미를 가진다.
// (기존 동기식 write→read 반복은 큐가 한 번에 비워져 threshold가 무력화됐음)

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void async_inference(hailo_configured_network_group network_group,
                     std::vector<cv::Mat>& pre,   // 전처리 완료 이미지 (공유, latency 측정서 제외)
                     size_t output_count,
                     const char* model_name,
                     double& total_latency,
                     int& total_count,
                     long& thread_vol_ctx,
                     long& thread_nonvol_ctx) {
    size_t input_count = 1;
    hailo_input_vstream_params_by_name_t input_params[1];
    hailo_output_vstream_params_by_name_t output_params[16];
    hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_UINT8, input_params, &input_count);
    hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_FLOAT32, output_params, &output_count);

    hailo_input_vstream input_vstreams[1];
    hailo_output_vstream output_vstreams[16];
    hailo_create_input_vstreams(network_group, input_params, input_count, input_vstreams);
    hailo_create_output_vstreams(network_group, output_params, output_count, output_vstreams);

    size_t N = pre.size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;

    // writer: 프레임을 큐에 밀어넣음. INPUT_FPS>0이면 그 속도로 제한(큐가 늘 꽉 차지 않게 함).
    std::thread writer([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        const double interval_ms = (INPUT_FPS > 0) ? (1000.0 / INPUT_FPS) : 0.0;
        double next_t = now_ms();
        for (size_t i = 0; i < N; i++) {
            if (INPUT_FPS > 0) {
                double t = now_ms();
                if (t < next_t)
                    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(next_t - t));
                next_t += interval_ms;
            }
            enq_ts[i] = now_ms();  // 전처리 제외, write 직전 시각
            hailo_vstream_write_raw_buffer(input_vstreams[0],
                pre[i].data, pre[i].total() * pre[i].elemSize());
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    // reader: 프레임별로 모든 출력 스트림을 읽고 수신완료 시각 기록
    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        size_t osz[16];
        std::vector<std::vector<uint8_t>> obuf(output_count);
        for (size_t j = 0; j < output_count; j++) {
            hailo_get_output_vstream_frame_size(output_vstreams[j], &osz[j]);
            obuf[j].resize(osz[j]);
        }
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < output_count; j++)
                hailo_vstream_read_raw_buffer(output_vstreams[j], obuf[j].data(), osz[j]);
            deq_ts[i] = now_ms();  // 프레임 i의 모든 출력 수신 완료 시각
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

    // latency = 출력수신 - 입력투입 (전처리 제외; 큐대기 + 추론 + 전송 포함 → threshold/timeout 효과 반영)
    double sum = 0; int c = 0;
    for (size_t i = 0; i < N; i++)
        if (deq_ts[i] > enq_ts[i]) { sum += (deq_ts[i] - enq_ts[i]); c++; }
    total_latency = c > 0 ? sum / c : -1;
    total_count = c;
    thread_vol_ctx    = w_vol + r_vol;      // writer+reader 두 스레드 합산
    thread_nonvol_ctx = w_nonvol + r_nonvol;

    hailo_release_input_vstreams(input_vstreams, input_count);
    hailo_release_output_vstreams(output_vstreams, output_count);

    std::lock_guard<std::mutex> lock(print_mutex);
    printf("[%s] 완료: 평균 Latency=%.2f ms, 총 %d장 (async)\n", model_name, total_latency, total_count);
}

// ========== CSV 저장 ==========

void save_csv(const char* csv_path,
              int run_id,
              double det_lat, double seg_lat, double pose_lat,
              double cpu, double mem,
              long vol_ctx, long nonvol_ctx) {
    std::ifstream check(csv_path);
    bool write_header = !check.good();
    check.close();

    std::ofstream f(csv_path, std::ios::app);
    if (write_header)
        f << "run_id,use_det,use_seg,use_pose,batch,"
          << "threshold_det,threshold_seg,threshold_pose,timeout_ms,"
          << "priority_det,priority_seg,priority_pose,"
          << "det_latency_ms,seg_latency_ms,pose_latency_ms,"
          << "cpu_percent,mem_percent,"
          << "voluntary_ctx_switches,nonvoluntary_ctx_switches,npu_percent\n";

    auto val = [](int use, int priority) -> std::string {
        return use ? std::to_string(priority) : "None";
    };
    auto lat = [](double l) -> std::string {
        return l >= 0 ? std::to_string(l) : "None";
    };
    // 모델별 threshold: PER_MODEL면 독립값, EQ_PRIORITY면 priority(0→THRESHOLD), 아니면 THRESHOLD. 비활성=None
    auto thr = [](int use, int priority, int thr_def) -> std::string {
        if (!use) return "None";
        if (THRESHOLD_PER_MODEL) return std::to_string(thr_def);
        if (THRESHOLD_EQ_PRIORITY) return std::to_string(priority == 0 ? THRESHOLD : priority);
        return std::to_string(THRESHOLD);
    };

    f << run_id << ","
      << USE_DET << "," << USE_SEG << "," << USE_POSE << ","
      << BATCH_SIZE << ","
      << thr(USE_DET, PRIORITY_DET, THRESHOLD_DET) << ","
      << thr(USE_SEG, PRIORITY_SEG, THRESHOLD_SEG) << ","
      << thr(USE_POSE, PRIORITY_POSE, THRESHOLD_POSE) << ","
      << TIMEOUT_MS << ","
      << val(USE_DET, PRIORITY_DET) << ","
      << val(USE_SEG, PRIORITY_SEG) << ","
      << val(USE_POSE, PRIORITY_POSE) << ","
      << lat(det_lat) << ","
      << lat(seg_lat) << ","
      << lat(pose_lat) << ","
      << cpu << "," << mem << ","
      << vol_ctx << "," << nonvol_ctx << ",\n";  // npu_percent는 parse_npu_log.py가 채움
    f.close();
    printf("결과 저장: %s\n", csv_path);
}

// ========== main ==========

int main(int argc, char* argv[]) {
    // argv[1]: run_id (없으면 1)
    // argv[2]: csv_path (없으면 기본값)
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    const char* csv_path = (argc > 2) ? argv[2] : CSV_PATH_DEFAULT;

    pid_t my_pid = getpid();
    printf("PID: %d, Run ID: %d\n", my_pid, run_id);

    hailo_status status;
    hailo_vdevice vdevice;
    hailo_vdevice_params_t vdevice_params;

    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    status = hailo_create_vdevice(&vdevice_params, &vdevice);
    if (status != HAILO_SUCCESS) { printf("VDevice 생성 실패\n"); return 1; }
    printf("VDevice 생성 성공!\n");

    struct ModelConfig {
        const char* hef_path;
        const char* name;
        size_t output_count;
        int priority;
        int threshold;
        bool active;
    };

    ModelConfig models[] = {
        {DET_HEF,  "Detection",    DET_OUTPUT_COUNT,  PRIORITY_DET,  THRESHOLD_DET,  (bool)USE_DET},
        {SEG_HEF,  "Segmentation", SEG_OUTPUT_COUNT,  PRIORITY_SEG,  THRESHOLD_SEG,  (bool)USE_SEG},
        {POSE_HEF, "Pose",         POSE_OUTPUT_COUNT, PRIORITY_POSE, THRESHOLD_POSE, (bool)USE_POSE},
    };

    hailo_hef hefs[3];
    hailo_configured_network_group network_groups[3];
    int active_count = 0;

    for (int i = 0; i < 3; i++) {
        if (!models[i].active) continue;
        status = hailo_create_hef_file(&hefs[i], models[i].hef_path);
        if (status != HAILO_SUCCESS) { printf("%s HEF 로드 실패\n", models[i].name); return 1; }

        hailo_configure_params_t configure_params;
        hailo_init_configure_params_by_vdevice(hefs[i], vdevice, &configure_params);
        for (size_t j = 0; j < configure_params.network_group_params_count; j++)
            configure_params.network_group_params[j].batch_size = BATCH_SIZE;

        size_t ng_size = 1;
        status = hailo_configure_vdevice(vdevice, hefs[i], &configure_params, &network_groups[i], &ng_size);
        if (status != HAILO_SUCCESS) { printf("%s 네트워크 설정 실패\n", models[i].name); return 1; }

        // threshold 결정:
        //  THRESHOLD_PER_MODEL=1 -> 모델별 독립 threshold (priority×threshold 조합 실험)
        //  THRESHOLD_EQ_PRIORITY=1 -> threshold=priority (priority=0이면 1)
        //  둘 다 0 -> 전역 THRESHOLD 고정
        uint32_t thr;
        if (THRESHOLD_PER_MODEL)
            thr = (uint32_t)models[i].threshold;
        else if (THRESHOLD_EQ_PRIORITY)
            thr = (models[i].priority == 0) ? (uint32_t)THRESHOLD : (uint32_t)models[i].priority;
        else
            thr = (uint32_t)THRESHOLD;
        // 스케줄러 파라미터 설정. HailoRT엔 getter가 없으므로, setter의 반환 status로
        // 적용 여부를 확인한다(HAILO_SUCCESS=적용됨). 성공/실패를 모두 명시적으로 로그.
        hailo_status st_thr = hailo_set_scheduler_threshold(network_groups[i], thr, NULL);
        hailo_status st_to  = hailo_set_scheduler_timeout(network_groups[i], TIMEOUT_MS, NULL);
        hailo_status st_pri = hailo_set_scheduler_priority(network_groups[i], models[i].priority, NULL);
        printf("  [적용확인] %s: threshold=%u [%s], timeout=%d [%s], priority=%d [%s]\n",
            models[i].name,
            thr,               (st_thr == HAILO_SUCCESS ? "OK" : "실패"),
            TIMEOUT_MS,        (st_to  == HAILO_SUCCESS ? "OK" : "실패"),
            models[i].priority,(st_pri == HAILO_SUCCESS ? "OK" : "실패"));
        if (st_thr != HAILO_SUCCESS || st_to != HAILO_SUCCESS || st_pri != HAILO_SUCCESS)
            printf("  [경고] %s 일부 파라미터 적용 실패! (status thr=%d to=%d pri=%d)\n",
                   models[i].name, st_thr, st_to, st_pri);

        printf("%s 설정 완료! (batch=%d, threshold=%u, timeout=%d, priority=%d)\n",
            models[i].name, BATCH_SIZE, thr, TIMEOUT_MS, models[i].priority);
        active_count++;
    }

    std::vector<std::string> images = get_image_files(IMG_DIR);
    // 검증 이미지 수 제한 (NUM_IMAGES > 0 일 때만)
    if (NUM_IMAGES > 0 && images.size() > (size_t)NUM_IMAGES)
        images.resize(NUM_IMAGES);
    printf("\n사용 이미지 수: %zu장, 활성 모델: %d개\n", images.size(), active_count);

    // ── 전처리 일괄 수행 (latency 측정 대상에서 제외; 모든 모델이 공유) ──
    // 비동기 writer가 큐를 빠르게 채우려면 전처리가 병목이 되면 안 되므로 미리 처리.
    std::vector<cv::Mat> pre;
    pre.reserve(images.size());
    for (auto& path : images) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        cv::Mat lb = letterbox(img, 640);
        cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
        pre.push_back(lb);
    }
    printf("전처리 완료: %zu장 (메모리 약 %.0f MB)\n\n",
           pre.size(), pre.size() * 640.0 * 640.0 * 3 / 1e6);

    // 측정 시작
    CpuStats cpu_start = read_cpu_stats();

    double latencies[3] = {-1, -1, -1};
    int counts[3] = {0, 0, 0};
    long vol_ctx_arr[3]    = {0, 0, 0};
    long nonvol_ctx_arr[3] = {0, 0, 0};

    // 모델별로 async_inference 실행 (각자 내부에서 writer/reader 스레드를 띄움)
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; i++) {
        if (!models[i].active) continue;
        threads.emplace_back(async_inference,
            network_groups[i], std::ref(pre),
            models[i].output_count, models[i].name,
            std::ref(latencies[i]), std::ref(counts[i]),
            std::ref(vol_ctx_arr[i]), std::ref(nonvol_ctx_arr[i]));
    }
    for (auto& t : threads) t.join();

    // 측정 종료
    CpuStats cpu_end = read_cpu_stats();

    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();
    // 각 추론 스레드가 측정한 context switch를 합산 (워커 스레드는 join 후 사라지므로
    // 메인 프로세스 status만으로는 추론 부하의 switch를 잡을 수 없음)
    long vol_ctx = 0, nonvol_ctx = 0;
    for (int i = 0; i < 3; i++) {
        if (!models[i].active) continue;
        vol_ctx    += vol_ctx_arr[i];
        nonvol_ctx += nonvol_ctx_arr[i];
    }

    printf("\n========== 실험 결과 ==========\n");
    printf("Run ID: %d\n", run_id);
    printf("모델: Det=%d, Seg=%d, Pose=%d\n", USE_DET, USE_SEG, USE_POSE);
    printf("Priority: Det=%d, Seg=%d, Pose=%d\n\n", PRIORITY_DET, PRIORITY_SEG, PRIORITY_POSE);
    if (USE_DET)  printf("Detection 평균 Latency: %.2f ms\n", latencies[0]);
    if (USE_SEG)  printf("Segmentation 평균 Latency: %.2f ms\n", latencies[1]);
    if (USE_POSE) printf("Pose 평균 Latency: %.2f ms\n", latencies[2]);
    printf("CPU: %.2f%%, MEM: %.2f%%\n", final_cpu, final_mem);
    printf("Context Switches - Voluntary: %ld, NonVoluntary: %ld\n", vol_ctx, nonvol_ctx);
    printf("================================\n");

    save_csv(csv_path, run_id, latencies[0], latencies[1], latencies[2],
             final_cpu, final_mem, vol_ctx, nonvol_ctx);

    for (int i = 0; i < 3; i++)
        if (models[i].active) hailo_release_hef(hefs[i]);
    hailo_release_vdevice(vdevice);
    return 0;
}
