#!/usr/bin/env python3
"""Download a remote file via base64 over exec (SFTP-free).
Usage: ssh_download.py <host_key> <remote> <local>

npu-rpi5(H10 호스트)는 SFTP 서브시스템이 열려 있지 않아 paramiko의 sftp.put/get이
FileNotFoundError로 떨어진다. 업로드용 ssh_upload.py와 짝이 되는 다운로드 쪽 도구.
"""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_run import connect

key, remote, local = sys.argv[1], sys.argv[2], sys.argv[3]
c, user = connect(key)
_, sout, serr = c.exec_command("base64 -w0 '%s'" % remote, timeout=300)
data = sout.read().decode()
rc = sout.channel.recv_exit_status()
if rc != 0:
    sys.exit("[FAIL] %s: %s" % (remote, serr.read().decode().strip()))
d = os.path.dirname(os.path.abspath(local))
if d:
    os.makedirs(d, exist_ok=True)
with open(local, "wb") as f:
    f.write(base64.b64decode(data))
print("[OK] get %s@%s:%s -> %s (%d bytes)" % (user, key, remote, local, os.path.getsize(local)))
c.close()
