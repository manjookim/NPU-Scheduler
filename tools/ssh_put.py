#!/usr/bin/env python3
"""Upload/download files over SFTP. Usage:
   ssh_put.py <host_key> put <local> <remote>
   ssh_put.py <host_key> get <remote> <local>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_run import connect

key, mode, a, b = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c, user = connect(key)
sftp = c.open_sftp()
if mode == "put":
    sftp.put(a, b)
    print(f"[OK] put {a} -> {user}@{key}:{b} ({os.path.getsize(a)} bytes)")
else:
    sftp.get(a, b)
    print(f"[OK] get {user}@{key}:{a} -> {b} ({os.path.getsize(b)} bytes)")
sftp.close(); c.close()
