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

// ========== 파라미터 설정 ==========
#define BATCH_SIZE      1
#define THRESHOLD       1
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
#define IMG_DIR  "/home/rpi1/datasets/coco/val2017/"
#define CSV_PATH "/home/rpi1/hailo_cpp_test/results_all.csv"

#define DET_OUTPUT_COUNT  1
#define SEG_OUTPUT_COUNT  10
#define POSE_OUTPUT_COUNT 9

std::mutex print_mutex;

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

long read_context_switches() {
    std::ifstream f("/proc/stat");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("ctxt") != std::string::npos) {
            long ctxt = 0;
            sscanf(line.c_str(), "ctxt %ld", &ctxt);
            return ctxt;
        }
    }
    return 0;
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
    return files;
}

void inference_thread(hailo_configured_network_group network_group,
                      std::vector<std::string>& images,
                      size_t output_count,
                      const char* model_name,
                      double& total_latency,
                      int& total_count) {
    size_t input_count = 1;
    hailo_input_vstream_params_by_name_t input_params[1];
    hailo_output_vstream_params_by_name_t output_params[16];

    hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_UINT8, input_params, &input_count);
    hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_FLOAT32, output_params, &output_count);

    hailo_input_vstream input_vstreams[1];
    hailo_output_vstream output_vstreams[16];

    hailo_create_input_vstreams(network_group, input_params, input_count, input_vstreams);
    hailo_create_output_vstreams(network_group, output_params, output_count, output_vstreams);

    double latency_sum = 0;
    int count = 0;
    int batch_count = 0;

    for (size_t idx = 0; idx + BATCH_SIZE <= images.size(); idx += BATCH_SIZE) {
        std::vector<cv::Mat> batch_imgs;
        for (int b = 0; b < BATCH_SIZE; b++) {
            cv::Mat img = cv::imread(images[idx + b]);
            if (img.empty()) continue;
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(640, 640));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
            batch_imgs.push_back(resized);
        }
        if ((int)batch_imgs.size() != BATCH_SIZE) continue;

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t b = 0; b < batch_imgs.size(); b++)
            hailo_vstream_write_raw_buffer(input_vstreams[0],
                batch_imgs[b].data, batch_imgs[b].total() * batch_imgs[b].elemSize());

        for (size_t b = 0; b < batch_imgs.size(); b++)
            for (size_t i = 0; i < output_count; i++) {
                size_t output_size = 0;
                hailo_get_output_vstream_frame_size(output_vstreams[i], &output_size);
                uint8_t* buf = (uint8_t*)malloc(output_size);
                hailo_vstream_read_raw_buffer(output_vstreams[i], buf, output_size);
                free(buf);
            }

        auto end = std::chrono::high_resolution_clock::now();
        latency_sum += std::chrono::duration<double, std::milli>(end - start).count() / BATCH_SIZE;
        count += BATCH_SIZE;
        batch_count++;

        if (batch_count % 500 == 0) {
            std::lock_guard<std::mutex> lock(print_mutex);
            printf("[%s] 진행 중: %d / %zu장\n", model_name, count, images.size());
        }
    }

    total_latency = batch_count > 0 ? latency_sum / batch_count : -1;
    total_count = count;

    hailo_release_input_vstreams(input_vstreams, input_count);
    hailo_release_output_vstreams(output_vstreams, output_count);

    std::lock_guard<std::mutex> lock(print_mutex);
    printf("[%s] 완료: 평균 Latency=%.2f ms, 총 %d장\n", model_name, total_latency, total_count);
}

void save_csv(double det_lat, double seg_lat, double pose_lat,
              double cpu, double mem, long ctx) {
    std::ifstream check(CSV_PATH);
    bool write_header = !check.good();
    check.close();

    std::ofstream f(CSV_PATH, std::ios::app);
    if (write_header)
        f << "use_det,use_seg,use_pose,batch,threshold,timeout_ms,"
          << "priority_det,priority_seg,priority_pose,"
          << "det_latency_ms,seg_latency_ms,pose_latency_ms,"
          << "cpu_percent,mem_percent,context_switches,npu_percent\n";

    auto val = [](int use, int priority) -> std::string {
        return use ? std::to_string(priority) : "None";
    };
    auto lat = [](double l) -> std::string {
        return l >= 0 ? std::to_string(l) : "None";
    };

    f << USE_DET << "," << USE_SEG << "," << USE_POSE << ","
      << BATCH_SIZE << "," << THRESHOLD << "," << TIMEOUT_MS << ","
      << val(USE_DET, PRIORITY_DET) << ","
      << val(USE_SEG, PRIORITY_SEG) << ","
      << val(USE_POSE, PRIORITY_POSE) << ","
      << lat(det_lat) << ","
      << lat(seg_lat) << ","
      << lat(pose_lat) << ","
      << cpu << "," << mem << "," << ctx << ",\n";
    f.close();
    printf("결과 저장: %s\n", CSV_PATH);
}

int main() {
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
        bool active;
    };

    ModelConfig models[] = {
        {DET_HEF,  "Detection",    DET_OUTPUT_COUNT,  PRIORITY_DET,  (bool)USE_DET},
        {SEG_HEF,  "Segmentation", SEG_OUTPUT_COUNT,  PRIORITY_SEG,  (bool)USE_SEG},
        {POSE_HEF, "Pose",         POSE_OUTPUT_COUNT, PRIORITY_POSE, (bool)USE_POSE},
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

        hailo_set_scheduler_threshold(network_groups[i], THRESHOLD, NULL);
        hailo_set_scheduler_timeout(network_groups[i], TIMEOUT_MS, NULL);
        hailo_set_scheduler_priority(network_groups[i], models[i].priority, NULL);

        printf("%s 설정 완료! (batch=%d, threshold=%d, timeout=%d, priority=%d)\n",
            models[i].name, BATCH_SIZE, THRESHOLD, TIMEOUT_MS, models[i].priority);
        active_count++;
    }

    std::vector<std::string> images = get_image_files(IMG_DIR);
    printf("\n총 이미지 수: %zu장, 활성 모델: %d개\n\n", images.size(), active_count);

    long ctx_start = read_context_switches();
    CpuStats cpu_start = read_cpu_stats();

    double latencies[3] = {-1, -1, -1};
    int counts[3] = {0, 0, 0};

    std::vector<std::thread> threads;
    for (int i = 0; i < 3; i++) {
        if (!models[i].active) continue;
        threads.emplace_back(inference_thread,
            network_groups[i], std::ref(images),
            models[i].output_count, models[i].name,
            std::ref(latencies[i]), std::ref(counts[i]));
    }
    for (auto& t : threads) t.join();

    long ctx_end = read_context_switches();
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);
    double final_mem = read_mem_usage();
    long ctx_total = ctx_end - ctx_start;

    printf("\n========== 실험 결과 ==========\n");
    printf("모델: Det=%d, Seg=%d, Pose=%d\n", USE_DET, USE_SEG, USE_POSE);
    printf("Priority: Det=%d, Seg=%d, Pose=%d\n\n", PRIORITY_DET, PRIORITY_SEG, PRIORITY_POSE);
    if (USE_DET)  printf("Detection 평균 Latency: %.2f ms\n", latencies[0]);
    if (USE_SEG)  printf("Segmentation 평균 Latency: %.2f ms\n", latencies[1]);
    if (USE_POSE) printf("Pose 평균 Latency: %.2f ms\n", latencies[2]);
    printf("CPU: %.2f%%, MEM: %.2f%%, Context Switch: %ld\n", final_cpu, final_mem, ctx_total);
    printf("================================\n");

    save_csv(latencies[0], latencies[1], latencies[2], final_cpu, final_mem, ctx_total);

    for (int i = 0; i < 3; i++)
        if (models[i].active) hailo_release_hef(hefs[i]);
    hailo_release_vdevice(vdevice);
    return 0;
}