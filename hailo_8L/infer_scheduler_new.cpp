/*
2-2. INPUT_FPS -> INPUT_FPS1,2,3 
2-3. ADD l99_latency
2-5. ADD hw latency 

*/ 

#include "hailo/hailort.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <algorithm>
#include <unistd.h>

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

// [2026-08-26] 우리 infer_scheduler.cpp의 "동일조건 비교" 실험(3모델, 600장)과 맞추기 위해
// USE_SEG/USE_POSE도 켜고 NUM_IMAGES도 600으로 맞춤. 원래 값(USE_SEG=0, USE_POSE=0, NUM_IMAGES=0)
// 이 필요하면 이 블록만 되돌릴 것.
#define USE_DET    1
#define USE_SEG    1
#define USE_POSE   1

#define NUM_IMAGES      600

#define INPUT_FPS_1   0
#define INPUT_FPS_2   0
#define INPUT_FPS_3   0

#define U_QUEUE_SIZE  HAILO_DEFAULT_VSTREAM_QUEUE_SIZE

// [2026-08-26] 원래 경로(/app/tappas/rpi2/...)는 다른 호스트(rpi2, tappas 도커) 기준이라
// 이 호스트(npu-rpi1)에는 없음 — npu-rpi1의 실제 경로로 교체(계정명 rpi1->npu-rpi1 변경 반영).
#define DET_HEF  "/home/npu-rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/npu-rpi1/hailo-rpi5-examples/resources/yolov8s_seg.hef"
#define POSE_HEF "/home/npu-rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"

#define IMG_DIR  "/home/npu-rpi1/datasets/sampled_val2017/"

cv::Mat letterbox(const cv::Mat& img, int target_size = 640) {
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
    return out;
}

std::vector<std::string> get_image_files(const char* dir_path) {
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

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct ModelResult {
    double avg_latency_ms = -1;
    double l99_latency_ms = -1;
    int frame_count = 0;
};

struct ModelConfig {
    const char* hef_path;
    const char* name;
    int priority;
    int threshold;
    int timeout_ms;
    int batch;
    bool active;
    int input_fps;
};

void run_model_async(const char* model_name,
                     std::vector<InputVStream>& inputs,
                     std::vector<OutputVStream>& outputs,
                     const std::vector<std::string>& images,
                     int input_fps,
                     ModelResult& result)
{
    if (inputs.empty() || outputs.empty()) {
        std::cerr << "[" << model_name << "] 입력/출력 vstream 없음, 스킵" << std::endl;
        return;
    }

    size_t expected = inputs[0].get_frame_size();
    size_t N = images.size();
    std::vector<double> enq_ts(N, 0.0), deq_ts(N, 0.0);
    hailo_status write_status = HAILO_SUCCESS, read_status = HAILO_SUCCESS;

    const size_t QUEUE_SIZE = HAILO_DEFAULT_VSTREAM_QUEUE_SIZE;
    std::vector<std::vector<uint8_t>> input_buffers(
        QUEUE_SIZE, std::vector<uint8_t>(expected));

    std::thread writer([&]() {
        const double interval_ms = (input_fps > 0) ? (1000.0 / input_fps) : 0.0;
        double next_t = now_ms();

        for (size_t i = 0; i < images.size(); i++) {
            cv::Mat img = cv::imread(images[i]);
            cv::Mat lb;

            if (img.empty()) {
                std::cerr << "[경고] 이미지 로드 실패: " << images[i] << " (빈 프레임으로 대체함)\n";
                lb = cv::Mat::zeros(640, 640, CV_8UC3); // 640x640 검은색 빈 이미지 생성
            } else {
                lb = letterbox(img, 640);
            }
            cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);

            if (input_fps > 0) {
                double t = now_ms();
                if (t < next_t)
                    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(next_t - t));
                next_t += interval_ms;
            }
            auto& buf = input_buffers[i % QUEUE_SIZE];
            std::memcpy(buf.data(), lb.data, expected);

            enq_ts[i] = now_ms();
            hailo_status status;
            do {
                status = inputs[0].write(MemoryView(buf.data(), buf.size()));
            } while (status == HAILO_TIMEOUT);

            if (HAILO_SUCCESS != status) { write_status = status; }
        }
    });

    std::thread reader([&]() {
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
        }
    });

    writer.join();
    reader.join();

    if (HAILO_SUCCESS != write_status || HAILO_SUCCESS != read_status) {
        std::cerr << "[" << model_name << "] [경고] 추론 중 오류 (write=" << write_status
                   << ", read=" << read_status << ") — 아래 latency는 왜곡됐을 수 있음" << std::endl;
    }

    double sum = 0; int c = 0;
    std::vector<double> latencies;
    for (size_t i = 0; i < N; i++)
        if (deq_ts[i] > enq_ts[i]) { 
            sum += (deq_ts[i] - enq_ts[i]); c++; 
            latencies.push_back(deq_ts[i] - enq_ts[i]);
        }

    if (!latencies.empty()){
        std::sort(latencies.begin(), latencies.end());

        size_t idx99 = (size_t)(latencies.size() * 0.99);
        if (idx99 >= latencies.size()) idx99 = latencies.size() -1;

        result.l99_latency_ms = latencies[idx99];
    }


    result.avg_latency_ms = c > 0 ? sum / c : -1;
    result.frame_count = c;

    //std::printf("[%s] 완료: 평균 Latency=%.2f ms, %d장, (async, INPUT_FPS=%d)\n",
    //            model_name, result.avg_latency_ms, result.frame_count,  input_fps);
}

static std::string dtos(double v) {           // 음수(-1)=미측정/비활성 → NaN
    if (v < 0) return "NaN";
    std::ostringstream os; os << v; return os.str();
}

void save_csv(const std::string& csv_path, int run_id,
              const std::vector<ModelConfig>& models,   // [0]=Det, [1]=Seg, [2]=Pose 고정 순서
              const ModelResult results[3])
{
    static const char* HEADER =
        "run_id,use_det,use_seg,use_pose,batch,"
        "threshold_det,threshold_seg,threshold_pose,timeout_ms,"
        "priority_det,priority_seg,priority_pose,"
        "det_latency_ms,seg_latency_ms,pose_latency_ms,";  

    bool need_header = true;
    {
        std::ifstream chk(csv_path);
        if (chk.good() && chk.peek() != std::ifstream::traits_type::eof())
            need_header = false;
    }
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "[CSV] Open Failed!: " << csv_path << std::endl;
        return;
    }
    if (need_header) f << HEADER << "\n";

    double det_lat  = models[0].active ? results[0].avg_latency_ms : -1;
    double seg_lat  = models[1].active ? results[1].avg_latency_ms : -1;
    double pose_lat = models[2].active ? results[2].avg_latency_ms : -1;

    std::ostringstream row;
    row << run_id << ','
        << (models[0].active ? 1 : 0) << ','
        << (models[1].active ? 1 : 0) << ','
        << (models[2].active ? 1 : 0) << ','
        << models[0].batch << ','                                  // batch (모델 통일값)
        << models[0].threshold << ',' << models[1].threshold << ',' << models[2].threshold << ','
        << models[0].timeout_ms << ','                             // timeout_ms (실험에서 모델 통일)
        << models[0].priority << ',' << models[1].priority << ',' << models[2].priority << ','
        << dtos(det_lat) << ',' << dtos(seg_lat) << ',' << dtos(pose_lat) << ',';
        
    f << row.str() << "\n";
    f.close();

    std::printf("[CSV] 저장: %s (run_id=%d, inference metric result, HRTT value : NaN)\n",
                csv_path.c_str(), run_id);
}

// ========================= main =========================

int main(int argc, char* argv[])
{
    int run_id = (argc > 1) ? atoi(argv[1]) : 1;
    std::string csv_path = (argc > 2) ? argv[2] : "";   // argv[2] 있으면 CSV 저장

    pid_t my_pid = getpid();
    std::printf("PID: %d, Run ID: %d\n", my_pid, run_id);

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

    std::vector<ModelConfig> models = {
        {DET_HEF,  "Detection",    PRIORITY_DET,  THRESHOLD_DET,  TIMEOUT_DET_MS,  BATCH_DET,  (bool)USE_DET, INPUT_FPS_1},
        {SEG_HEF,  "Segmentation", PRIORITY_SEG,  THRESHOLD_SEG,  TIMEOUT_SEG_MS,  BATCH_SEG,  (bool)USE_SEG, INPUT_FPS_2},
        {POSE_HEF, "Pose",         PRIORITY_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, BATCH_POSE, (bool)USE_POSE, INPUT_FPS_3},
    };

    std::vector<std::shared_ptr<ConfiguredNetworkGroup>> network_groups;
    std::vector<int> active_model_idx;  // network_groups[k] <-> models[active_model_idx[k]]

    for (size_t i = 0; i < models.size(); i++) {
        auto& m = models[i];
        if (!m.active) continue;

        auto hef_exp = Hef::create(m.hef_path);
        if (!hef_exp) { std::cerr << m.name << " HEF 로드 실패" << std::endl; return (int)hef_exp.status(); }
        auto hef = hef_exp.release();


        auto cfg_exp = vdevice->create_configure_params(hef);
        if (!cfg_exp) { std::cerr << m.name << " configure params 실패" << std::endl; return (int)cfg_exp.status(); }
        auto cfg = cfg_exp.value();
        for (auto& ng_param : cfg) {
            ng_param.second.batch_size = m.batch;
            ng_param.second.power_mode = HAILO_POWER_MODE_ULTRA_PERFORMANCE;
            ng_param.second.latency = HAILO_LATENCY_MEASURE;
        }

        auto ngs_exp = vdevice->configure(hef, cfg);
        if (!ngs_exp) { std::cerr << m.name << " configure 실패" << std::endl; return (int)ngs_exp.status(); }
        auto network_group = ngs_exp.value()[0];

        if (m.threshold > m.batch)
            std::printf("  [경고] %s: threshold(%d) > batch(%d) — set_scheduler_threshold가 실패할 것으로 예상됨. "
                        "THRESHOLD_* <= BATCH_*로 맞출 것.\n", m.name, m.threshold, m.batch);

        //auto st_thr = network_group->set_scheduler_threshold((uint32_t)m.threshold);
        //auto st_to  = network_group->set_scheduler_timeout(std::chrono::milliseconds(m.timeout_ms));
        //auto st_pri = network_group->set_scheduler_priority((uint8_t)m.priority);
        //std::printf("  [적용확인] %-13s: batch=%d, threshold=%d [%s], timeout=%dms [%s], priority=%d [%s]\n",
        //    m.name, m.batch,
        //    m.threshold,  (st_thr == HAILO_SUCCESS ? "OK" : "실패"),
        //    m.timeout_ms, (st_to  == HAILO_SUCCESS ? "OK" : "실패"),
        //    m.priority,   (st_pri == HAILO_SUCCESS ? "OK" : "실패"));
        //if (st_thr != HAILO_SUCCESS || st_to != HAILO_SUCCESS || st_pri != HAILO_SUCCESS)
        //    std::printf("  [경고] %s 일부 파라미터 적용 실패! (thr=%d to=%d pri=%d)\n",
        //                m.name, (int)st_thr, (int)st_to, (int)st_pri);

        network_groups.push_back(network_group);
        active_model_idx.push_back((int)i);
    }

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
    std::printf("사용 이미지 수: %zu장 (경로: %s)\n", images.size(), IMG_DIR);

    const uint32_t VSTREAM_TIMEOUT_MS = 300000;  // 5분 — starvation 대기 여유
    std::vector<std::pair<std::vector<InputVStream>, std::vector<OutputVStream>>> vstreams_per_ng;
    for (auto& ng : network_groups) {
        auto in_params  = ng->make_input_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS,
                              U_QUEUE_SIZE); 
                              //HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        auto out_params = ng->make_output_vstream_params(false, HAILO_FORMAT_TYPE_AUTO,
                              VSTREAM_TIMEOUT_MS,
                              U_QUEUE_SIZE);
                              // HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_params)  { std::cerr << "input vstream params 실패, status="  << in_params.status()  << std::endl; return (int)in_params.status(); }
        if (!out_params) { std::cerr << "output vstream params 실패, status=" << out_params.status() << std::endl; return (int)out_params.status(); }

        auto in_vs  = ng->create_input_vstreams(in_params.value());
        auto out_vs = ng->create_output_vstreams(out_params.value());
        if (!in_vs)  { std::cerr << "input vstream 생성 실패, status="  << in_vs.status()  << std::endl; return (int)in_vs.status(); }
        if (!out_vs) { std::cerr << "output vstream 생성 실패, status=" << out_vs.status() << std::endl; return (int)out_vs.status(); }

        vstreams_per_ng.emplace_back(std::make_pair(in_vs.release(), out_vs.release()));
    }

    ModelResult results[3];  // index: Detection=0, Segmentation=1, Pose=2 (models 배열과 동일 순서)
    
    double total_start_ms = now_ms();
    
    std::vector<std::thread> threads;
    
    for (size_t k = 0; k < vstreams_per_ng.size(); k++) {
        int mi = active_model_idx[k];
        threads.emplace_back(run_model_async, models[mi].name,
            std::ref(vstreams_per_ng[k].first), std::ref(vstreams_per_ng[k].second),
            std::cref(images), models[mi].input_fps, std::ref(results[mi]));
    }
    for (auto& t : threads) t.join();

    double total_end_ms = now_ms();
    double total_time_ms = total_end_ms - total_start_ms;

    std::printf("\n================= 실험 결과==================\n", run_id);
    std::printf("Total Time :  %.2f s\n", total_time_ms/1000);
    std::printf("Throughput :  %.2f FPS\n\n", (images.size() / (total_time_ms / 1000.0)));

    std::printf("\n========== 하드웨어(HW) Latency 측정 결과 ==========\n");
    for (size_t k = 0; k < network_groups.size(); k++) {
        int mi = active_model_idx[k];
        auto hw_lat_exp = network_groups[k]->get_latency_measurement();
        
        if (hw_lat_exp) {
            double hw_avg_ms = hw_lat_exp.value().avg_hw_latency.count() / 1000000.0;
            std::printf("[%s] 순수 NPU 연산 시간(HW Avg): %.3f ms\n", models[mi].name, hw_avg_ms);
            
        } else {
            std::printf("[%s] HW Latency 측정 실패 (Error: %d)\n", models[mi].name, hw_lat_exp.status());
        }
    }

    if (USE_DET)  std::printf("Detection    : latency=%.2fms, L99=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[0].avg_latency_ms, results[0].l99_latency_ms, results[0].frame_count, BATCH_DET, THRESHOLD_DET, TIMEOUT_DET_MS, PRIORITY_DET);
    if (USE_SEG)  std::printf("Segmentation : latency=%.2fms, L99=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[1].avg_latency_ms, results[1].l99_latency_ms, results[1].frame_count, BATCH_SEG, THRESHOLD_SEG, TIMEOUT_SEG_MS, PRIORITY_SEG);
    if (USE_POSE) std::printf("Pose         : latency=%.2fms, L99=%.2fms, %d장, batch=%d, threshold=%d, timeout=%dms, priority=%d\n",
                              results[2].avg_latency_ms, results[2].l99_latency_ms, results[2].frame_count, BATCH_POSE, THRESHOLD_POSE, TIMEOUT_POSE_MS, PRIORITY_POSE);

    if (!csv_path.empty())
        save_csv(csv_path, run_id, models, results);

    return HAILO_SUCCESS;
}
