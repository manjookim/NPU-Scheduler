#!/usr/bin/env python3
"""Password-auth SSH helper (paramiko). Usage: ssh_run.py <host_key> "<command>" """
import sys, paramiko

HOSTS = {
    "rpi1":  dict(host="155.230.16.157", port=40021, users=["npu-rpi1", "rpi1"], pw="0000"),
    "rpi5":  dict(host="155.230.16.157", port=40020, users=["npu-rpi5"],         pw="0000"),
    "gpu":   dict(host="155.230.27.126", port=40000, users=["oslab"],            pw="1"),
    "gpu2":  dict(host="155.230.16.154", port=40027, users=["gpu-master"],       pw="0"),
    "gpu3":  dict(host="155.230.16.157", port=40026, users=["gpu"],              pw="1"),
}

def connect(key):
    cfg = HOSTS[key]
    last = None
    for user in cfg["users"]:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(cfg["host"], port=cfg["port"], username=user, password=cfg["pw"],
                      timeout=15, banner_timeout=20, auth_timeout=20,
                      look_for_keys=False, allow_agent=False)
            return c, user
        except Exception as e:
            last = f"{user}: {type(e).__name__}: {e}"
    raise SystemExit(f"[FAIL] {key} -> {last}")

if __name__ == "__main__":
    key, cmd = sys.argv[1], sys.argv[2]
    c, user = connect(key)
    print(f"[OK] connected {key} as {user}", flush=True)
    _, out, err = c.exec_command(cmd, timeout=300, get_pty=False)
    print(out.read().decode("utf-8", "replace"), end="")
    e = err.read().decode("utf-8", "replace")
    if e.strip():
        print("--- stderr ---\n" + e, end="")
    print(f"[exit={out.channel.recv_exit_status()}]")
    c.close()
