# HRTT 통합판 3개 스크립트를 보드로 전송 (PowerShell)
scp -P 40024 `
  "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hailo_8\scripts\run_v5det_seg_pose.sh" `
  "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hailo_8\scripts\run_v5det_cpu_seg_pose.sh" `
  "C:\Users\sset0\jungmin-claude\StudentExperiment\NPUscheduler\hailo_8\scripts\run_v5det_npu_vs_cpu_compare.sh" `
  rpi4@155.230.16.157:~/hailo_cpp_test/scripts/
