#!/usr/bin/env python3
"""Upload a local file to a remote host via base64 over exec (SFTP-free).
Usage: ssh_upload.py <host_key> <local> <remote>"""
import sys, os, base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_run import connect

key, local, remote = sys.argv[1], sys.argv[2], sys.argv[3]
data = base64.b64encode(open(local, "rb").read()).decode()
c, user = connect(key)
rdir = os.path.dirname(remote)
cmd = f"mkdir -p '{rdir}' && cat > '{remote}.b64' && base64 -d '{remote}.b64' > '{remote}' && rm -f '{remote}.b64' && wc -c '{remote}'"
sin, sout, serr = c.exec_command(cmd, timeout=180)
# write in chunks
for i in range(0, len(data), 32000):
    sin.write(data[i:i+32000])
sin.channel.shutdown_write()
out = sout.read().decode(); err = serr.read().decode()
rc = sout.channel.recv_exit_status()
print(out.strip() or err.strip(), f"[exit={rc}]")
c.close()
sys.exit(rc)
