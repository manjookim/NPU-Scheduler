# -*- coding: utf-8 -*-
"""npu-rpi5(Hailo-10H)에서 default_workload 스윕 결과(csv/traces/logs)를 조회·다운로드.

비밀번호는 파일에 넣지 않는다. 환경변수 HAILO10H_PW 로 전달할 것.

  HAILO10H_PW='...' python tools/fetch_h10_results.py --list
  HAILO10H_PW='...' python tools/fetch_h10_results.py --fetch
"""
import argparse
import os
import posixpath
import stat
import sys

import paramiko

HOST = '155.230.16.157'
PORT = 40020
USER = 'npu-rpi5'
REMOTE_ROOT = '/home/npu-rpi5/hailo10h_sched_exp1'
LOCAL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'hailo_10h', 'experiments', '2026-08-24_det_seg_pose_workload_exp1')
SUBDIRS = ['csv', 'traces', 'logs']


def connect():
    pw = os.environ.get('HAILO10H_PW')
    if not pw:
        sys.exit('환경변수 HAILO10H_PW 가 비어 있습니다.')
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pw, timeout=20)
    return cli


def run(cli, cmd):
    _, out, err = cli.exec_command(cmd)
    return out.read().decode('utf-8', 'replace'), err.read().decode('utf-8', 'replace')


def do_list(cli):
    print('=== hostname / 실험 디렉토리 ===')
    print(run(cli, 'hostname; ls -la %s 2>&1' % REMOTE_ROOT)[0])
    for d in SUBDIRS:
        print('=== %s/ ===' % d)
        print(run(cli, 'ls -la %s/%s 2>&1 | head -40' % (REMOTE_ROOT, d))[0])
    print('=== 집계 CSV 미리보기 ===')
    csvp = '%s/csv/results_det_seg_pose_workload.csv' % REMOTE_ROOT
    out, _ = run(cli, 'wc -l %s 2>&1; echo ---; head -3 %s 2>&1' % (csvp, csvp))
    print(out)
    print('=== .hrtt (0바이트 확인용) ===')
    print(run(cli, 'find %s -name "*.hrtt" -printf "%%s\\t%%p\\n" 2>/dev/null | head -20' % REMOTE_ROOT)[0])


def fetch_dir(sftp, remote, local):
    try:
        entries = sftp.listdir_attr(remote)
    except IOError:
        print('  [건너뜀] 원격에 없음: %s' % remote)
        return 0
    os.makedirs(local, exist_ok=True)
    n = 0
    for e in sorted(entries, key=lambda x: x.filename):
        rp = posixpath.join(remote, e.filename)
        lp = os.path.join(local, e.filename)
        if stat.S_ISDIR(e.st_mode):
            n += fetch_dir(sftp, rp, lp)
        else:
            sftp.get(rp, lp)
            print('  %8d B  %s' % (e.st_size, os.path.relpath(lp, LOCAL_ROOT)))
            n += 1
    return n


def do_fetch(cli):
    sftp = cli.open_sftp()
    total = 0
    for d in SUBDIRS:
        print('=== %s/ ===' % d)
        total += fetch_dir(sftp, posixpath.join(REMOTE_ROOT, d),
                           os.path.join(LOCAL_ROOT, d))
    sftp.close()
    print('\n총 %d개 파일 -> %s' % (total, LOCAL_ROOT))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true', help='원격 상태만 조회')
    ap.add_argument('--fetch', action='store_true', help='csv/traces/logs 다운로드')
    a = ap.parse_args()
    if not (a.list or a.fetch):
        ap.error('--list 또는 --fetch 중 하나를 지정할 것')
    c = connect()
    try:
        if a.list:
            do_list(c)
        if a.fetch:
            do_fetch(c)
    finally:
        c.close()
