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

// ========== 파라미터 설정 (여기만 바꾸면 됩니다) ==========
#define BATCH_SIZE     1
#define THRESHOLD      0
#define TIMEOUT_MS     0
#define PRIORITY       0
// ==========================================================

#define DET_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_h8l.hef"
#define SEG_HEF  "/home/rpi1/hailo-rpi5-examples/resources/yolov5n_seg_h8l.hef"
#define POSE_HEF "/home/rpi1/hailo-rpi5-examples/resources/yolov8s_pose_h8l.hef"
#define IMG_DIR  "/home/rpi1/datasets/coco/val2017/"
#define CSV_PATH "/home/rpi1/hailo_cpp_test/results.csv"

#define DET_OUTPUT_COUNT  1
#define SEG_OUTPUT_COUNT  4
#define POSE_OUTPUT_COUNT 9

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

hailo_status run_inference_batch(hailo_configured_network_group network_group,
                                  std::vector<cv::Mat>& batch_imgs,
                                  size_t output_count,
                                  double& latency_ms) {
    hailo_status status;
    size_t input_count = 1;
    size_t actual_batch = batch_imgs.size();

    hailo_input_vstream_params_by_name_t input_params[1];
    hailo_output_vstream_params_by_name_t output_params[16];

    status = hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_UINT8, input_params, &input_count);
    if (status != HAILO_SUCCESS) return status;

    status = hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_FLOAT32, output_params, &output_count);
    if (status != HAILO_SUCCESS) return status;

    hailo_input_vstream input_vstreams[1];
    hailo_output_vstream output_vstreams[16];

    status = hailo_create_input_vstreams(network_group, input_params, input_count, input_vstreams);
    if (status != HAILO_SUCCESS) return status;

    status = hailo_create_output_vstreams(network_group, output_params, output_count, output_vstreams);
    if (status != HAILO_SUCCESS) return status;

    auto start = std::chrono::high_resolution_clock::now();

    // BATCH_SIZE만큼 이미지를 순서대로 전송
    for (size_t b = 0; b < actual_batch; b++) {
        status = hailo_vstream_write_raw_buffer(input_vstreams[0],
            batch_imgs[b].data,
            batch_imgs[b].total() * batch_imgs[b].elemSize());
        if (status != HAILO_SUCCESS) return status;
    }

    // BATCH_SIZE만큼 출력 읽기
    for (size_t b = 0; b < actual_batch; b++) {
        for (size_t i = 0; i < output_count; i++) {
            size_t output_size = 0;
            hailo_get_output_vstream_frame_size(output_vstreams[i], &output_size);
            uint8_t* buf = (uint8_t*)malloc(output_size);
            hailo_vstream_read_raw_buffer(output_vstreams[i], buf, output_size);
            free(buf);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    latency_ms = std::chrono::duration<double, std::milli>(end - start).count() / actual_batch;

    hailo_release_input_vstreams(input_vstreams, input_count);
    hailo_release_output_vstreams(output_vstreams, output_count);

    return HAILO_SUCCESS;
}

void save_csv(double det_lat, double seg_lat, double pose_lat,
              double cpu, double mem, long ctx) {
    std::ifstream check(CSV_PATH);
    bool write_header = !check.good();
    check.close();

    std::ofstream f(CSV_PATH, std::ios::app);
    if (write_header)
        f << "batch,threshold,timeout_ms,priority,det_latency_ms,seg_latency_ms,pose_latency_ms,cpu_percent,mem_percent,context_switches,npu_percent\n";

    f << BATCH_SIZE << ","
      << THRESHOLD << ","
      << TIMEOUT_MS << ","
      << PRIORITY << ","
      << det_lat << ","
      << seg_lat << ","
      << pose_lat << ","
      << cpu << ","
      << mem << ","
      << ctx << ",\n";
    f.close();
    printf("결과가 %s 에 저장됐습니다!\n", CSV_PATH);
}

int main() {
    hailo_status status;
    hailo_vdevice vdevice;
    hailo_vdevice_params_t vdevice_params;

    status = hailo_init_vdevice_params(&vdevice_params);
    if (status != HAILO_SUCCESS) { printf("vdevice params 초기화 실패: %d\n", status); return 1; }

    vdevice_params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    status = hailo_create_vdevice(&vdevice_params, &vdevice);
    if (status != HAILO_SUCCESS) { printf("VDevice 생성 실패: %d\n", status); return 1; }
    printf("VDevice 생성 성공! (스케줄러: Round Robin)\n");

    const char* hef_paths[3] = {DET_HEF, SEG_HEF, POSE_HEF};
    const char* model_names[3] = {"Detection", "Segmentation", "Pose"};
    size_t output_counts[3] = {DET_OUTPUT_COUNT, SEG_OUTPUT_COUNT, POSE_OUTPUT_COUNT};
    hailo_hef hefs[3];
    hailo_configured_network_group network_groups[3];

    for (int i = 0; i < 3; i++) {
        status = hailo_create_hef_file(&hefs[i], hef_paths[i]);
        if (status != HAILO_SUCCESS) { printf("%s HEF 로드 실패: %d\n", model_names[i], status); return 1; }

        hailo_configure_params_t configure_params;
        status = hailo_init_configure_params_by_vdevice(hefs[i], vdevice, &configure_params);
        if (status != HAILO_SUCCESS) { printf("%s configure params 실패: %d\n", model_names[i], status); return 1; }

        for (size_t j = 0; j < configure_params.network_group_params_count; j++) {
            configure_params.network_group_params[j].batch_size = BATCH_SIZE;
        }

        size_t network_group_size = 1;
        status = hailo_configure_vdevice(vdevice, hefs[i], &configure_params, &network_groups[i], &network_group_size);
        if (status != HAILO_SUCCESS) { printf("%s 네트워크 설정 실패: %d\n", model_names[i], status); return 1; }

        hailo_set_scheduler_threshold(network_groups[i], THRESHOLD, NULL);
        hailo_set_scheduler_timeout(network_groups[i], TIMEOUT_MS, NULL);
        hailo_set_scheduler_priority(network_groups[i], PRIORITY, NULL);

        printf("%s 모델 설정 완료! (batch=%d, threshold=%d, timeout=%dms, priority=%d)\n",
            model_names[i], BATCH_SIZE, THRESHOLD, TIMEOUT_MS, PRIORITY);
    }

    std::vector<std::string> images = get_image_files(IMG_DIR);
    printf("\n총 이미지 수: %zu장\n\n", images.size());

    double total_latency[3] = {0, 0, 0};
    double cpu_usage_sum = 0;
    double mem_usage_sum = 0;
    long ctx_start = read_context_switches();
    CpuStats cpu_start = read_cpu_stats();
    int count = 0;
    int batch_count = 0;

    // BATCH_SIZE만큼 묶어서 추론
    for (size_t idx = 0; idx + BATCH_SIZE <= images.size(); idx += BATCH_SIZE) {
        // 배치 이미지 준비
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

        // 3개 모델 추론
        for (int i = 0; i < 3; i++) {
            double latency = 0;
            status = run_inference_batch(network_groups[i], batch_imgs, output_counts[i], latency);
            if (status != HAILO_SUCCESS) { printf("추론 실패: %d\n", status); return 1; }
            total_latency[i] += latency;
        }

        CpuStats cpu_now = read_cpu_stats();
        cpu_usage_sum += calc_cpu_usage(cpu_start, cpu_now);
        mem_usage_sum += read_mem_usage();
        count += BATCH_SIZE;
        batch_count++;

        if (batch_count % (100 / BATCH_SIZE + 1) == 0)
            printf("진행 중: %d / %zu장\n", count, images.size());
    }

    long ctx_end = read_context_switches();
    CpuStats cpu_end = read_cpu_stats();
    double final_cpu = calc_cpu_usage(cpu_start, cpu_end);

    double det_avg = total_latency[0] / batch_count;
    double seg_avg = total_latency[1] / batch_count;
    double pose_avg = total_latency[2] / batch_count;
    double mem_avg = mem_usage_sum / batch_count;
    long ctx_total = ctx_end - ctx_start;

    printf("\n========== 실험 결과 ==========\n");
    printf("파라미터: batch=%d, threshold=%d, timeout=%dms, priority=%d\n\n",
        BATCH_SIZE, THRESHOLD, TIMEOUT_MS, PRIORITY);
    printf("총 추론 이미지: %d장 (%d 배치)\n\n", count, batch_count);
    printf("Detection 평균 Latency: %.2f ms\n", det_avg);
    printf("Segmentation 평균 Latency: %.2f ms\n", seg_avg);
    printf("Pose 평균 Latency: %.2f ms\n", pose_avg);
    printf("\n평균 CPU 사용률: %.2f%%\n", final_cpu);
    printf("평균 MEM 사용률: %.2f%%\n", mem_avg);
    printf("총 Context Switching: %ld\n", ctx_total);
    printf("================================\n");

    save_csv(det_avg, seg_avg, pose_avg, final_cpu, mem_avg, ctx_total);

    for (int i = 0; i < 3; i++) hailo_release_hef(hefs[i]);
    hailo_release_vdevice(vdevice);
    return 0;
}
