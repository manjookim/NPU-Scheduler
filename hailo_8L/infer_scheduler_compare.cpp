//제 코드에 있는건 +로 표기하겠으며 반대로 없는 것은 -로 표기하겠습니다.


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
#include <map>
#include <utility>
#include <memory>

// [원본: infer_scheduler.cpp] 후처리(decode_det/pose/seg), LetterboxMeta, TensorView 등
// (397줄 분량이라 이 파일엔 인라인하지 않음 — 실제 내용은 postprocess_8l.hpp 참고)
#include "postprocess_8l.hpp"

using namespace hailort;

#define BATCH_DET       1
#define BATCH_SEG       1
#define BATCH_POSE      1

#define THRESHOLD_DET   1
#define THRESHOLD_SEG   1
#define THRESHOLD_POSE  1

#define TIMEOUT_DET_MS   0
#define TIMEOUT_SEG_MS   0
#define TIMEOUT_POSE_MS  0

#define PRIORITY_DET    15
#define PRIORITY_SEG    15
#define PRIORITY_POSE   15

// 세 모델 모두 활성화 (YOLOv8 기준)
#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

//++++++++++++++++++++++++++//
// [진단용] 1=Pose/Seg 디코딩+NMS 수행(정상), 0=디코딩 스킵.
#define ENABLE_POSTPROCESS  1

//+++++++++++++++++++++++++//
// [진단용] 1=write() 블로킹시간 계측(처음 40프레임만). 평소엔 0.
#define DEBUG_WRITE_TIMING  0

//-----------------------------//
//저는 모든 입력데이터를 동일한 fps로 고정하였습니다.
#define INPUT_FPS       0

// 사용할 검증 이미지 수 (0 = IMG_DIR의 전체 이미지 사용)
#define NUM_IMAGES      600

// HEF 경로 (Raspberry Pi 5, hailo-rpi5-examples 리소스) — 세 모델 모두 YOLOv8 기반
#define DET_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"

#define IMG_DIR  "/home/rpi1/datasets/sampled_val2017/"

//++++++++++++++++++++++++++//
//cmd에 모델별 시작상태를 출력하기 위해 mutex를 사용하였으며, 시간 측정 구간 외에 실행하였으며, 각 케이스별 최초1회만 실행
std::mutex print_mutex;


//LetterboxMeta는 후처리시 box의 원본 좌표 복원을 위해서 사용되었습니다.
inline cv::Mat letterbox(const cv::Mat& img, int target_size = 640, LetterboxMeta* meta_out = nullptr) {
    int orig_h = img.rows, orig_w = img.cols;
    float scale = std::min((float)target_size / orig_h, (float)target_size / orig_w);
    int new_h = (int)(orig_h * scale);
    int new_w = (int)(orig_w * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    int pad_top    = (target_size - new_h) / 2;
    int pad_bottom = target_size - new_h - pad_top;
    int pad_left   = (target_size - new_w) / 2;
    int pad_right  = target_size - new_w - pad_left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right,
                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    //
    //추론후 검출 박스를 원본 형태에 맞게 역변환 해야합니다. 아래는 해당 역변환을 도와주는 값들을 미리 저장하는 코드입니다.
    if (meta_out) {
        meta_out->scale = scale;
        meta_out->pad_top = pad_top;
        meta_out->pad_left = pad_left;
        meta_out->orig_w = orig_w;
        meta_out->orig_h = orig_h;
    }
    return out;
}


inline std::vector<std::string> get_image_files(const char* dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path);
    if (!dir) return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name.size() > 4 &&
            (name.find(".jpg") != std::string::npos || name.find(".JPG") != std::string::npos))
            files.push_back(std::string(dir_path) + name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());  // 실험 재현성을 위해 정렬(항상 같은 부분집합 사용)
    return files;
}

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct ModelResult {
    double avg_latency_ms = -1;
    int frame_count = 0;
    long vol_ctx = 0;
    long nonvol_ctx = 0;
    double total_time_s = -1;       // 이 모델이 모든 입력을 처리하는 데 걸린 전체 시간(초)
    double avg_preprocess_ms = -1;  // 프레임당 평균 전처리(imread+letterbox) 시간 — 모델별 독립 측정
    double avg_postprocess_ms = -1; // 프레임당 평균 후처리(디코딩+NMS) 시간
    double avg_total_time_ms = -1;  // 장당 전체 시간 = 전처리 + latency + 후처리
    //---------------
    //다만 저희는 tail_latency는 추가하지 않았습니다.
};

//+++++++++++++++++++++++++++
//후처리 파이프 라인에서 모델에 따른 후처리 과정 선택
enum class ModelKind { DET, SEG, POSE };


struct ModelConfig {
    const char* hef_path;
    const char* name;
    int priority;
    int threshold;
    uint32_t timeout_ms;  // HailoRT 내부적으로 uint32_t를 사용합니다. 다만 조교님의 실험에서 int값을 넘는 timeout는 없기 떄문에 사실상 큰 차이점은 아닙니다.
    int batch;
    bool active;
    ModelKind kind; //++++ 이후 코드에서 설명드리겠지만, 저희는 모델에 따른 후처리가 다르기 때문에 후처리 모드를 설정하기 위한 변수값입니다. 
    int img_size = 640; //++++ 저희가 v5로도 실험했기 때문에  size가 다른 경우를 대비해서 만든 변수입니다.
    //---- 저희는 모델별 fps를 다르게주지 않아서 input_fps 변수값은 없습니다.
};

.
enum class OutRole { OTHER, POSE_BOX, POSE_SCORE, POSE_KPTS, SEG_BOX, SEG_CLS, SEG_COEFF, SEG_PROTO };

struct OutMeta {
    OutRole role = OutRole::OTHER;
    int h = 0, w = 0, c = 0;
    int stride = 0;  // img_size / h (정사각 640 입력 가정)
    // Detection(NMS-by-class 출력) 전용 — get_info().nms_shape에서 얻음. Pose/Seg는 0으로 둠.
    int nms_number_of_classes = 0;
    int nms_max_bboxes_per_class = 0;
};

// ============================================================================
// [원본: output_classify.hpp] classify_outputs() — output vstream role 분류
// ============================================================================
inline std::vector<OutMeta> classify_outputs(ModelKind kind, std::vector<OutputVStream>& outputs, int img_size = 640) {
    std::vector<OutMeta> metas(outputs.size());
    if (kind == ModelKind::DET) {
        // Detection은 보통 NMS-by-class 출력 vstream 1개. decode_det()가 버퍼를 파싱하는 데
        // 필요한 클래스 수/클래스당 최대 검출 수만 얻는다.
        for (size_t j = 0; j < outputs.size(); j++) {
            const auto& info = outputs[j].get_info();
            metas[j].nms_number_of_classes = (int)info.nms_shape.number_of_classes;
            metas[j].nms_max_bboxes_per_class = (int)info.nms_shape.max_bboxes_per_class;
        }
        return metas;
    }

    for (size_t j = 0; j < outputs.size(); j++) {
        const auto& info = outputs[j].get_info();
        int h = (int)info.shape.height, w = (int)info.shape.width, c = (int)info.shape.features;
        metas[j].h = h; metas[j].w = w; metas[j].c = c;
        metas[j].stride = (h > 0) ? (img_size / h) : 0;

        if (kind == ModelKind::POSE) {
            if (c == 64) metas[j].role = OutRole::POSE_BOX;
            else if (c == 1) metas[j].role = OutRole::POSE_SCORE;
            else if (c == 51) metas[j].role = OutRole::POSE_KPTS;
        } else if (kind == ModelKind::SEG) {
            if (c == 64) metas[j].role = OutRole::SEG_BOX;
            else if (c == 80) metas[j].role = OutRole::SEG_CLS;
            else if (c == 32) metas[j].role = (h == 160 && w == 160) ? OutRole::SEG_PROTO : OutRole::SEG_COEFF;
        }
    }
    return metas;
}

// ============================================================================
// [원본: sys_monitor.hpp] CPU / 메모리 / 컨텍스트 스위치 측정 유틸
// (조교님 코드엔 대응 섹션 없음 — new는 이 지표들을 측정하지 않음)
// ============================================================================
struct CpuStats {
    long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
};

inline CpuStats read_cpu_stats() {
    CpuStats s;
    std::ifstream f("/proc/stat");
    std::string line;
    std::getline(f, line);
    sscanf(line.c_str(), "cpu %ld %ld %ld %ld %ld %ld %ld",
           &s.user, &s.nice, &s.system, &s.idle, &s.iowait, &s.irq, &s.softirq);
    return s;
}

inline double calc_cpu_usage(const CpuStats& s1, const CpuStats& s2) {
    long idle1 = s1.idle + s1.iowait;
    long idle2 = s2.idle + s2.iowait;
    long total1 = s1.user + s1.nice + s1.system + s1.idle + s1.iowait + s1.irq + s1.softirq;
    long total2 = s2.user + s2.nice + s2.system + s2.idle + s2.iowait + s2.irq + s2.softirq;
    long dt = total2 - total1;
    if (dt <= 0) return 0.0;
    return 100.0 * (1.0 - (double)(idle2 - idle1) / (double)dt);
}

inline double read_mem_usage() {
    std::ifstream f("/proc/meminfo");
    long total = 0, available = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("MemTotal:") != std::string::npos)
            sscanf(line.c_str(), "MemTotal: %ld kB", &total);
        if (line.find("MemAvailable:") != std::string::npos)
            sscanf(line.c_str(), "MemAvailable: %ld kB", &available);
    }
    if (total <= 0) return 0.0;
    return 100.0 * (1.0 - (double)available / (double)total);
}

struct CtxSwitches { long voluntary = 0, nonvoluntary = 0; };

// 호출한 스레드 자신의 context switch 읽기 (/proc/thread-self/status, Linux 3.17+).
inline CtxSwitches read_thread_ctx_switches() {
    CtxSwitches cs;
    std::ifstream f("/proc/thread-self/status");
    std::string line;
    long v;
    while (std::getline(f, line)) {
        if (sscanf(line.c_str(), "nonvoluntary_ctxt_switches: %ld", &v) == 1)
            cs.nonvoluntary = v;
        else if (sscanf(line.c_str(), "voluntary_ctxt_switches: %ld", &v) == 1)
            cs.voluntary = v;
    }
    return cs;
}

// ============================================================================
// [원본: model_setup.hpp] configure_models() — HEF 로드 -> configure(batch) ->
// 스케줄러 파라미터(threshold/timeout/priority) 설정
// (조교님 코드는 이 블록을 main()에 인라인하되, 스케줄러 파라미터 3줄은 주석 처리되어 있음)
// ============================================================================
inline hailo_status configure_models(std::unique_ptr<VDevice>& vdevice,
                              std::vector<ModelConfig>& models,
                              std::vector<std::shared_ptr<ConfiguredNetworkGroup>>& network_groups,
                              std::vector<int>& active_model_idx)
{
    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;

        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) { std::cerr << m.name << " HEF 로드 실패" << std::endl; return hef_exp.status(); }
        auto hef = hef_exp.release();

        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) { std::cerr << m.name << " configure params 실패" << std::endl; return cfg_exp.status(); }
        auto cfg = cfg_exp.value();
        for (auto& ng_param : cfg) {
            ng_param.second.batch_size = m.batch;
            ng_param.second.power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
        }

        auto ngs_exp = vdevice->configure(hef, cfg);
        if (!ngs_exp) { std::cerr << m.name << " configure 실패" << std::endl; return ngs_exp.status(); }
        auto network_group = ngs_exp.value()[0];

        // 실기 확인된 제약: threshold는 batch_size 이하여야 함.
        if (m.threshold > m.batch)
            std::printf("  [경고] %s: threshold(%d) > batch(%d) — set_scheduler_threshold가 실패할 것으로 예상됨. "
                        "THRESHOLD_* <= BATCH_*로 맞출 것.\n", m.name, m.threshold, m.batch);

        // 스케줄러 파라미터 설정 (실제로 적용됨 — 조교님 코드는 이 3줄이 주석 처리되어 있음)
        auto st_thr = network_group->set_scheduler_threshold((uint32_t)m.threshold);
        auto st_to  = network_group->set_scheduler_timeout(std::chrono::milliseconds(m.timeout_ms));
        auto st_pri = network_group->set_scheduler_priority((uint8_t)m.priority);
        std::printf("  [적용확인] %-13s: batch=%d, threshold=%d [%s], timeout=%ums [%s], priority=%d [%s]\n",
            m.name, m.batch,
            m.threshold,  (st_thr == HAILO_SUCCESS ? "OK" : "실패"),
            m.timeout_ms, (st_to  == HAILO_SUCCESS ? "OK" : "실패"),
            m.priority,   (st_pri == HAILO_SUCCESS ? "OK" : "실패"));
        if (st_thr != HAILO_SUCCESS || st_to != HAILO_SUCCESS || st_pri != HAILO_SUCCESS)
            std::printf("  [경고] %s 일부 파라미터 적용 실패! (thr=%d to=%d pri=%d)\n",
                        m.name, (int)st_thr, (int)st_to, (int)st_pri);

        network_groups.push_back(network_group);
        active_model_idx.push_back((int)i);
    }
    return HAILO_SUCCESS;
}

// ============================================================================
// [원본: model_setup.hpp] create_all_vstreams() — vstream 생성
// (조교님 코드는 이 블록도 main()에 인라인, output 포맷은 AUTO로 둠)
// ============================================================================
inline hailo_status create_all_vstreams(
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>>& network_groups,
    const std::vector<ModelConfig>& models,
    const std::vector<int>& active_model_idx,
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>>& vstreams_per_ng,
    std::vector<std::vector<OutMeta>>& out_meta_per_ng)
{
    // [중요] vstream 기본 timeout은 10초다. 우선순위가 낮은 모델은 굶주려(starvation) 10초를
    // 넘길 수 있으므로 크게 잡아 write/read가 HAILO_TIMEOUT으로 죽지 않게 방지.
    const uint32_t VSTREAM_TIMEOUT_MS = 300000;  // 5분
    for (size_t k = 0; k < network_groups.size(); k++) {
        auto& ng = network_groups[k];
        ModelKind kind = models[active_model_idx[k]].kind;

        auto in_params  = ng->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        // 세 모델 모두 FLOAT32로 통일 — decode_det()이 NMS-by-class 버퍼를 float 배열로
        // 가정하고 파싱하므로 AUTO에 맡기지 않고 명시적으로 FLOAT32를 요청해야 함.
        hailo_format_type_t out_fmt_type = HAILO_FORMAT_TYPE_FLOAT32;
        auto out_params = ng->make_output_vstream_params(false, out_fmt_type,
                              VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_params)  { std::cerr << "input vstream params 실패, status="  << in_params.status()  << std::endl; return in_params.status(); }
        if (!out_params) { std::cerr << "output vstream params 실패, status=" << out_params.status() << std::endl; return out_params.status(); }

        // Segmentation/Pose는 raw tensor라서 order도 NHWC로 추가 고정한다.
        if (kind != ModelKind::DET) {
            for (auto& kv : out_params.value())
                kv.second.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
        }

#if DEBUG_WRITE_TIMING
        // [진단용] 큐 크기를 HailoRT 공식 accumulator로 직접 측정하기 위한 플래그.
        for (auto& kv : in_params.value())
            kv.second.pipeline_elements_stats_flags = HAILO_PIPELINE_ELEM_STATS_MEASURE_QUEUE_SIZE;
        for (auto& kv : out_params.value())
            kv.second.pipeline_elements_stats_flags = HAILO_PIPELINE_ELEM_STATS_MEASURE_QUEUE_SIZE;
#endif

        auto in_vs  = ng->create_input_vstreams(in_params.value());
        auto out_vs = ng->create_output_vstreams(out_params.value());
        if (!in_vs)  { std::cerr << "input vstream 생성 실패, status="  << in_vs.status()  << std::endl; return in_vs.status(); }
        if (!out_vs) { std::cerr << "output vstream 생성 실패, status=" << out_vs.status() << std::endl; return out_vs.status(); }

        auto out_vs_val = out_vs.release();
        out_meta_per_ng.push_back(classify_outputs(kind, out_vs_val));
        vstreams_per_ng.emplace_back(std::make_pair(in_vs.release(), std::move(out_vs_val)));
    }
    return HAILO_SUCCESS;
}

// ============================================================================
// [원본: model_runner.hpp] run_model_async() — 모델 1개 담당 writer/reader 스레드 쌍
// (조교님 코드와의 핵심 차이: 여기서는 reader가 실제 후처리(decode_det/pose/seg)를 수행함.
//  전처리/후처리 시간, 컨텍스트 스위치도 각각 별도로 측정한다.)
// ============================================================================
inline void run_model_async(const char* model_name,
                     ModelKind kind,
                     std::vector<InputVStream>& inputs,
                     std::vector<OutputVStream>& outputs,
                     const std::vector<OutMeta>& out_meta,
                     const std::vector<std::string>& images,
                     ModelResult& result,
                     int img_size = 640)
{
    if (inputs.empty() || outputs.empty()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    size_t N = images.size();
    size_t expected_frame_size = inputs[0].get_frame_size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    std::vector<LetterboxMeta> pre_meta(N);
    long w_vol = 0, w_nonvol = 0, r_vol = 0, r_nonvol = 0;
    hailo_status write_status = HAILO_SUCCESS, read_status = HAILO_SUCCESS;

    double prep_total_ms = 0.0;
    long prep_count = 0;

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

            // ── 전처리 (이 프레임만): imread -> letterbox -> BGR2RGB. 프레임당 소요시간 측정 ──
            double prep_t0 = now_ms();
            cv::Mat img = cv::imread(images[i]);
            cv::Mat lb;
            if (img.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "[" << model_name << "] [경고] 이미지 로드 실패: " << images[i]
                           << " (검은 화면으로 대체, 프레임 수/인덱스 정렬 유지)" << std::endl;
                lb = cv::Mat::zeros(img_size, img_size, CV_8UC3);
            } else {
                LetterboxMeta lm;
                lb = letterbox(img, img_size, &lm);
                cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
                pre_meta[i] = lm;
            }
            double prep_t1 = now_ms();
            prep_total_ms += (prep_t1 - prep_t0);
            prep_count++;

            if (i == 0 && lb.total() * lb.elemSize() != expected_frame_size) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "[" << model_name << "] [경고] 프레임 크기 불일치: 모델 기대="
                           << expected_frame_size << "B, 전처리 결과=" << (lb.total() * lb.elemSize())
                           << "B (letterbox 크기/채널 수를 모델 입력 shape에 맞게 조정할 것)" << std::endl;
            }

            enq_ts[i] = now_ms();
            hailo_status status;
            do {
                status = inputs[0].write(MemoryView(lb.data, lb.total() * lb.elemSize()));
            } while (status == HAILO_TIMEOUT);
            if (HAILO_SUCCESS != status) { write_status = status; }
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        w_vol = c1.voluntary - c0.voluntary; w_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    double pp_total_ms = 0.0;
    long pp_count = 0;

    std::thread reader([&]() {
        CtxSwitches c0 = read_thread_ctx_switches();
        std::vector<std::vector<uint8_t>> obuf(outputs.size());
        for (size_t j = 0; j < outputs.size(); j++)
            obuf[j].resize(outputs[j].get_frame_size());

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < outputs.size(); j++) {
                hailo_status status;
                do {
                    status = outputs[j].read(MemoryView(obuf[j].data(), obuf[j].size()));
                } while (status == HAILO_TIMEOUT);
                if (HAILO_SUCCESS != status) { read_status = status; }
            }
            deq_ts[i] = now_ms();  // 이 프레임의 모든 출력을 다 받은 시각

            // ── 후처리, 프레임당 소요시간 별도 측정 (조교님 코드엔 이 블록 자체가 없음) ──
            double pp_t0 = now_ms();
#if ENABLE_POSTPROCESS
            if (kind == ModelKind::DET) {
                if (!out_meta.empty() && out_meta[0].nms_number_of_classes > 0 && out_meta[0].nms_max_bboxes_per_class > 0) {
                    const LetterboxMeta* lm = (i < pre_meta.size()) ? &pre_meta[i] : nullptr;
                    std::vector<PPBox> det_dets = decode_det(reinterpret_cast<const float*>(obuf[0].data()),
                                                              out_meta[0].nms_number_of_classes,
                                                              out_meta[0].nms_max_bboxes_per_class,
                                                              0.0f, lm, img_size);
                    if (i == 0) {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::printf("  [디버그][%s] 첫 프레임: 클래스 %d개 x 클래스당 최대 %d개 파싱, 검출=%zu개",
                                    model_name, out_meta[0].nms_number_of_classes, out_meta[0].nms_max_bboxes_per_class, det_dets.size());
                        if (lm && !det_dets.empty())
                            std::printf(" (unpad 적용, 원본 %dx%d, 첫 검출 box=[%.1f,%.1f,%.1f,%.1f] coco_id=%d)",
                                        lm->orig_w, lm->orig_h, det_dets[0].x1, det_dets[0].y1, det_dets[0].x2, det_dets[0].y2, det_dets[0].class_id);
                        std::printf("\n");
                    }
                } else if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [경고][%s] nms_shape 정보를 못 얻어 후처리 파싱을 건너뜀\n", model_name);
                }
            } else if (kind == ModelKind::POSE) {
                std::map<std::pair<int,int>, PoseScaleTensors> groups;
                for (size_t j = 0; j < outputs.size(); j++) {
                    const auto& m = out_meta[j];
                    pp::TensorView tv{ reinterpret_cast<const float*>(obuf[j].data()), m.h, m.w, m.c };
                    auto& g = groups[{m.h, m.w}];
                    g.stride = m.stride;
                    if (m.role == OutRole::POSE_BOX) g.box = tv;
                    else if (m.role == OutRole::POSE_SCORE) g.score = tv;
                    else if (m.role == OutRole::POSE_KPTS) g.kpts = tv;
                }
                std::vector<PoseScaleTensors> scales;
                scales.reserve(groups.size());
                for (auto& kv : groups) scales.push_back(kv.second);
                size_t total_cells = 0;
                for (auto& sc : scales) total_cells += (size_t)sc.box.h * sc.box.w;
                size_t raw_cand = 0;
                std::vector<PoseDet> pose_dets = decode_pose(scales, 640, 0.3f, 0.45f, 0.5f, &raw_cand);
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [디버그][%s] 첫 프레임: 그리드 셀 총 %zu개 중 NMS 전 후보=%zu개, 최종 검출=%zu개\n",
                                model_name, total_cells, raw_cand, pose_dets.size());
                }
            } else if (kind == ModelKind::SEG) {
                std::map<std::pair<int,int>, SegScaleTensors> groups;
                pp::TensorView proto_tv;
                for (size_t j = 0; j < outputs.size(); j++) {
                    const auto& m = out_meta[j];
                    pp::TensorView tv{ reinterpret_cast<const float*>(obuf[j].data()), m.h, m.w, m.c };
                    if (m.role == OutRole::SEG_PROTO) { proto_tv = tv; continue; }
                    auto& g = groups[{m.h, m.w}];
                    g.stride = m.stride;
                    if (m.role == OutRole::SEG_BOX) g.box = tv;
                    else if (m.role == OutRole::SEG_CLS) g.cls = tv;
                    else if (m.role == OutRole::SEG_COEFF) g.coeff = tv;
                }
                std::vector<SegScaleTensors> scales;
                scales.reserve(groups.size());
                for (auto& kv : groups) scales.push_back(kv.second);
                size_t total_cells = 0;
                for (auto& sc : scales) total_cells += (size_t)sc.box.h * sc.box.w;
                size_t raw_cand = 0;
                std::vector<SegDet> seg_dets = decode_seg(scales, proto_tv, 640, 0.01f, 0.65f, 80, true, &raw_cand);
                if (i == 0) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::printf("  [디버그][%s] 첫 프레임: 그리드 셀 총 %zu개 중 NMS 전 후보=%zu개, 최종 검출=%zu개\n",
                                model_name, total_cells, raw_cand, seg_dets.size());
                }
            }
#endif  // ENABLE_POSTPROCESS
            double pp_t1 = now_ms();
            pp_total_ms += (pp_t1 - pp_t0);
            pp_count++;
        }
        CtxSwitches c1 = read_thread_ctx_switches();
        r_vol = c1.voluntary - c0.voluntary; r_nonvol = c1.nonvoluntary - c0.nonvoluntary;
    });

    writer.join();
    reader.join();

    result.avg_preprocess_ms = (prep_count > 0) ? (prep_total_ms / prep_count) : -1;
    result.avg_postprocess_ms = (pp_count > 0) ? (pp_total_ms / pp_count) : -1;

    if (HAILO_SUCCESS != write_status || HAILO_SUCCESS != read_status) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[" << model_name << "] [경고] 추론 중 오류 (write=" << write_status
                   << ", read=" << read_status << ") — 아래 latency는 왜곡됐을 수 있음" << std::endl;
    }

    double sum = 0; int c = 0;
    for (size_t i = 0; i < N; i++)
        if (deq_ts[i] > enq_ts[i]) { sum += (deq_ts[i] - enq_ts[i]); c++; }

    result.avg_latency_ms = c > 0 ? sum / c : -1;
    result.frame_count = c;
    result.vol_ctx = w_vol + r_vol;
    result.nonvol_ctx = w_nonvol + r_nonvol;

    double first_enq = 0, last_deq = 0;
    for (size_t i = 0; i < N; i++) {
        if (enq_ts[i] > 0 && (first_enq == 0 || enq_ts[i] < first_enq)) first_enq = enq_ts[i];
        if (deq_ts[i] > last_deq) last_deq = deq_ts[i];
    }
    result.total_time_s = (last_deq > first_enq) ? (last_deq - first_enq) / 1000.0 : -1;

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[%s] 완료: 전처리=%.2f ms, 평균 Latency=%.2f ms, 후처리=%.2f ms, %d장, 전체 추론시간=%.2f s (async, INPUT_FPS=%d)\n",
                model_name, result.avg_preprocess_ms, result.avg_latency_ms, result.avg_postprocess_ms, result.frame_count, result.total_time_s, INPUT_FPS);
}

inline std::string dtos(double v) {
    if (v < 0) return "NaN";
    std::ostringstream os; os << v; return os.str();
}


//제 코드에서 측정하는 값이 많아서 다를뿐, csv 저장함수는 거의 같습니다.
//다만 스레드와 mutex가 있는데, 모델별로 기록함과 동시에 습관적으로 걸어둔 것입니다. 측정 구간 외에 있으므로 결과에 영향은 없습니다.
inline void save_csv(const std::string& csv_path, int run_id,
              const std::vector<ModelConfig>& models,   // [0]=Det, [1]=Seg, [2]=Pose 고정 순서
              const ModelResult results[3],
              double cpu_percent, double mem_percent,
              long vol_ctx, long nonvol_ctx, double run_time_s)
{
    static const char* HEADER =
        "run_id,use_det,use_seg,use_pose,batch,"
        "threshold_det,threshold_seg,threshold_pose,timeout_ms,"
        "priority_det,priority_seg,priority_pose,"
        "det_latency_ms,seg_latency_ms,pose_latency_ms,"
        "cpu_percent,mem_percent,voluntary_ctx_switches,nonvoluntary_ctx_switches,"
        "npu_percent,switches_per_s,idle_time_pct,run_time_s,"
        "avg_fps_det,avg_latency_det,max_latency_det,activation_det,"
        "avg_fps_seg,avg_latency_seg,max_latency_seg,activation_seg,"
        "avg_fps_pose,avg_latency_pose,max_latency_pose,activation_pose,"
        "total_time_det_s,total_time_seg_s,total_time_pose_s,"
        "avg_preprocess_ms_det,avg_preprocess_ms_seg,avg_preprocess_ms_pose,"
        "postprocess_ms_det,postprocess_ms_seg,postprocess_ms_pose,"
        "total_time_ms_det,total_time_ms_seg,total_time_ms_pose,"
        "total_time_ms_nopp_det,total_time_ms_nopp_seg,total_time_ms_nopp_pose";

    bool need_header = true;
    {
        std::ifstream chk(csv_path);
        if (chk.good() && chk.peek() != std::ifstream::traits_type::eof())
            need_header = false;
    }
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "[CSV] 열기 실패: " << csv_path << std::endl;
        return;
    }
    if (need_header) f << HEADER << "\n";

    double det_lat  = models[0].active ? results[0].avg_latency_ms : -1;
    double seg_lat  = models[1].active ? results[1].avg_latency_ms : -1;
    double pose_lat = models[2].active ? results[2].avg_latency_ms : -1;
    double det_tot  = models[0].active ? results[0].total_time_s : -1;
    double seg_tot  = models[1].active ? results[1].total_time_s : -1;
    double pose_tot = models[2].active ? results[2].total_time_s : -1;
    double det_pp   = models[0].active ? results[0].avg_postprocess_ms : -1;
    double seg_pp   = models[1].active ? results[1].avg_postprocess_ms : -1;
    double pose_pp  = models[2].active ? results[2].avg_postprocess_ms : -1;
    double det_prep  = models[0].active ? results[0].avg_preprocess_ms : -1;
    double seg_prep  = models[1].active ? results[1].avg_preprocess_ms : -1;
    double pose_prep = models[2].active ? results[2].avg_preprocess_ms : -1;
    double det_ttl  = models[0].active ? results[0].avg_total_time_ms : -1;
    double seg_ttl  = models[1].active ? results[1].avg_total_time_ms : -1;
    double pose_ttl = models[2].active ? results[2].avg_total_time_ms : -1;
    auto nopp_total = [](double prep, double lat) {
        return (prep >= 0 && lat >= 0) ? (prep + lat) : -1;
    };
    double det_nopp  = models[0].active ? nopp_total(det_prep, det_lat)  : -1;
    double seg_nopp  = models[1].active ? nopp_total(seg_prep, seg_lat)  : -1;
    double pose_nopp = models[2].active ? nopp_total(pose_prep, pose_lat) : -1;

    std::ostringstream row;
    row << run_id << ','
        << (models[0].active ? 1 : 0) << ','
        << (models[1].active ? 1 : 0) << ','
        << (models[2].active ? 1 : 0) << ','
        << models[0].batch << ','
        << models[0].threshold << ',' << models[1].threshold << ',' << models[2].threshold << ','
        << models[0].timeout_ms << ','
        << models[0].priority << ',' << models[1].priority << ',' << models[2].priority << ','
        << dtos(det_lat) << ',' << dtos(seg_lat) << ',' << dtos(pose_lat) << ','
        << dtos(cpu_percent) << ',' << dtos(mem_percent) << ','
        << vol_ctx << ',' << nonvol_ctx << ','
        << "NaN" << ','                                            // npu_percent (HRTT 단계에서 채움)
        << "NaN" << ','                                            // switches_per_s
        << "NaN" << ','                                            // idle_time_pct
        << dtos(run_time_s) << ','
        << "NaN,NaN,NaN,NaN,"
        << "NaN,NaN,NaN,NaN,"
        << "NaN,NaN,NaN,NaN,"
        << dtos(det_tot) << ',' << dtos(seg_tot) << ',' << dtos(pose_tot) << ','
        << dtos(det_prep) << ',' << dtos(seg_prep) << ',' << dtos(pose_prep) << ','
        << dtos(det_pp) << ',' << dtos(seg_pp) << ',' << dtos(pose_pp) << ','
        << dtos(det_ttl) << ',' << dtos(seg_ttl) << ',' << dtos(pose_ttl) << ','
        << dtos(det_nopp) << ',' << dtos(seg_nopp) << ',' << dtos(pose_nopp);
    f << row.str() << "\n";
    f.close();

    std::lock_guard<std::mutex> lock(print_mutex);
    std::printf("[CSV] 저장: %s (run_id=%d, 추론 측정값 기록, HRTT값은 NaN)\n",
                csv_path.c_str(), run_id);
}

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";   // argv[2] 있으면 CSV 저장

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d\n", my_pid, run_id);

    // ── VDevice 생성 (스케줄러 Round-Robin) ──
    hailo_vdevice_params_t vdevice_params;
    hailo_init_vdevice_params(&vdevice_params);
    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    auto vdevice_exp = VDevice::create(vdevice_params);
    if (!vdevice_exp) {
        std::cerr << "VDevice 생성 실패, status=" << vdevice_exp.status() << std::endl;
        return (int)vdevice_exp.status();
    }
    auto vdevice = vdevice_exp.release();
    std::cout << "VDevice 생성 성공!" << std::endl;

    // 제코드는 모델별 input_fps는 없고 모델별 후처리 모드는 있습니다.
    std::vector<ModelConfig> models = {
        {DET_HEF,  "Detection",    PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  BATCH_DET,  (bool)USE_DET,  ModelKind::DET},
        {SEG_HEF,  "Segmentation", PRIORITY_SEG,  THRESHOLD_SEG,  TIMEOUT_SEG_MS,  BATCH_SEG,  (bool)USE_SEG,  ModelKind::SEG},
        {POSE_HEF, "Pose",         PRIORITY_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, BATCH_POSE, (bool)USE_POSE, ModelKind::POSE},
    };
    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
    std::vector<int> active_model_idx;
   
    //모델 별 설정 (모델선택/HEF파일 로드/ 배치크기 등), 다만 조교님께서는 main함수에서 처리하셨고, 저는 별도의 hpp파일에 정의 하였으며 로직은 전부 같습니다
    hailo_status cfg_status = configure_models(vdevice, models, network_groups, active_model_idx);
    if (cfg_status != HAILO_SUCCESS) return (int)cfg_status;

    if (network_groups.empty()) {
        std::cerr << "활성화된 모델이 없습니다 (USE_DET/USE_SEG/USE_POSE 확인)" << std::endl;
        return 1;
    }

    
    std::vector<std::string> images = get_image_files(IMG_DIR);
    if (images.empty()) {
        std::cerr << "[경고] IMG_DIR(" << IMG_DIR << ")에서 이미지를 찾지 못함. "
                  << "경로를 확인하고 #define IMG_DIR을 수정할 것." << std::endl;
        return 1;
    }
    if (NUM_IMAGES > 0 && images.size() > (size_t)NUM_IMAGES)
        images.resize(NUM_IMAGES);
    std::printf("사용 이미지 수: %zu장 (경로: %s)\n\n", images.size(), IMG_DIR);

    // ── vstream 생성 ──
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    std::vector<std::vector<OutMeta>> out_meta_per_ng; //후처리에 필요한 정보

    //아래 코드도 저희가 따로 만들었으며 내부 구조는 조교님께서 만드신 input/output stream과 같습니다.
    hailo_status vs_status = create_all_vstreams(network_groups, models, active_model_idx,
                                                  vstreams_per_ng, out_meta_per_ng);
    if (vs_status != HAILO_SUCCESS) return (int)vs_status;

    //++++ cpu 사용 시간
    CpuStats cpu_start = read_cpu_stats();
    double t_run_start = now_ms();

    ModelResult results[3];
    std::vector<std::thread> threads;

    for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
        int mi = active_model_idx[k];
        threads.emplace_back(run_model_async, models[mi].name, models[mi].kind,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(out_meta_per_ng[k]),
            std::cref(images), std::ref(results[mi]));
    }
    for (auto& t : threads) t.join();

    // 장당 전체시간(전처리-추론-후처리) = 이 모델 자신의 평균 전처리 + 평균 latency + 평균 후처리시간
    for (int i = 0; i < 3; i++) {
        if (models[i].active && results[i].avg_latency_ms >= 0) {
            double prep = (results[i].avg_preprocess_ms >= 0) ? results[i].avg_preprocess_ms : 0.0;
            double pp = (results[i].avg_postprocess_ms >= 0) ? results[i].avg_postprocess_ms : 0.0;
            results[i].avg_total_time_ms = prep + results[i].avg_latency_ms + pp;
        }
    }

    double run_time_s = (now_ms() - t_run_start) / 1000.0;
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();

    long vol_ctx = 0, nonvol_ctx = 0;
    for (auto& r : results) { vol_ctx += r.vol_ctx; nonvol_ctx += r.nonvol_ctx; }

    std::printf("\n========== 실험 결과 (Run ID: %d) ==========\n", run_id);
    if (USE_DET)  std::printf("Detection    : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[0].avg_preprocess_ms, results[0].avg_latency_ms, results[0].avg_postprocess_ms, results[0].avg_total_time_ms,
                              results[0].frame_count, BATCH_DET, THRESHOLD_DET, (uint32_t)TIMEOUT_DET_MS, PRIORITY_DET);
    if (USE_SEG)  std::printf("Segmentation : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[1].avg_preprocess_ms, results[1].avg_latency_ms, results[1].avg_postprocess_ms, results[1].avg_total_time_ms,
                              results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, (uint32_t)TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose         : 전처리=%.2fms, latency=%.2fms, 후처리=%.2fms, 전체=%.2fms, %d장, batch=%d, threshold=%d, timeout=%ums, priority=%d\n",
                              results[2].avg_preprocess_ms, results[2].avg_latency_ms, results[2].avg_postprocess_ms, results[2].avg_total_time_ms,
                              results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, (uint32_t)TIMEOUT_POSE_MS, PRIORITY_POSE);
    std::printf("CPU: %.2f%%, MEM: %.2f%%, Ctx Switch(vol/nonvol): %ld/%ld\n", final_cpu, final_mem, vol_ctx, nonvol_ctx);
    std::printf("================================================\n");

    // ── CSV 저장 (argv[2]로 경로가 주어졌을 때만) ──
    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results,
                 final_cpu, final_mem, vol_ctx, nonvol_ctx, run_time_s);

    return HAILO_SUCCESS;
}
