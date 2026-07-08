# -*- coding: utf-8 -*-
"""
verify_params.py
HRTT(.hrtt 또는 변환 .html)에서 HailoRT가 '실제로 적용한' 스케줄러 파라미터
(threshold/timeout/priority)를 core_op_set_value 트레이스로 확인.
파일명에 의도값이 있으면(예: ..._{PD}PD-..PP_{TD}TD-..TP..) 대조해 불일치를 표시.

사용법:
  python3 verify_params.py <파일 또는 폴더> [...]
  예) python3 verify_params.py hrtt/demo
"""
import sys, os, re, glob, base64
sys.path.insert(0, os.path.dirname(__file__))
import profiler_pb2

def load(path):
    if path.endswith(".hrtt"):
        p = profiler_pb2.ProtoProfiler(); p.ParseFromString(open(path, "rb").read()); return p
    # html
    m = re.search(r'PROTOBUF_BASE64_DATA_PLACEHOLDER="([A-Za-z0-9+/=]+)"', open(path, encoding="utf-8").read())
    if not m: return None
    p = profiler_pb2.ProtoProfiler(); p.ParseFromString(base64.b64decode(m.group(1))); return p

def lbl(name):
    n = name.lower()
    return "pose" if "pose" in n else ("seg" if "seg" in n else "det")

def intended(fname):
    """파일명에서 모델별 의도 priority/threshold 추출 (있으면)."""
    out = {}
    m = re.search(r'(\d)D-(\d)S-(\d)P_(\d+)PD-(\d+)PS-(\d+)PP(?:_(\d+)TD-(\d+)TS-(\d+)TP)?', os.path.basename(fname))
    if not m: return out
    g = m.groups()
    use = {'det': g[0]=='1', 'seg': g[1]=='1', 'pose': g[2]=='1'}
    pri = {'det': g[3], 'seg': g[4], 'pose': g[5]}
    thr = {'det': g[6], 'seg': g[7], 'pose': g[8]} if g[6] else None
    for m_ in ('det','seg','pose'):
        if use[m_]:
            out[m_] = {'priority': pri[m_], 'threshold': (thr[m_] if thr else None)}
    return out

def check(path):
    p = load(path)
    if p is None:
        print(f"  [!] protobuf 없음: {os.path.basename(path)}"); return
    names = {t.added_core_op.core_op_handle: t.added_core_op.core_op_name
             for t in p.added_trace if t.WhichOneof("trace") == "added_core_op"}
    applied = {}  # label -> {threshold,timeout,priority}
    for t in p.added_trace:
        if t.WhichOneof("trace") == "core_op_set_value":
            e = t.core_op_set_value; L = lbl(names.get(e.core_op_handle, ""))
            applied.setdefault(L, {})
            for f in ("threshold","timeout","priority"):
                if e.HasField(f): applied[L][f] = getattr(e, f)
    want = intended(path)
    print(f"■ {os.path.basename(path)}")
    for L in ('det','seg','pose'):
        if L not in applied and L not in want: continue
        a = applied.get(L, {})
        thr = a.get('threshold', '(미적용!)'); to = a.get('timeout', '(미적용!)'); pr = a.get('priority', '(미적용!)')
        flag = ""
        if L in want:
            w = want[L]
            if w['threshold'] is not None and str(a.get('threshold')) != w['threshold']:
                flag += f"  ⚠ threshold 의도={w['threshold']} / 적용={a.get('threshold','없음')}"
            if str(a.get('priority')) != w['priority']:
                flag += f"  ⚠ priority 의도={w['priority']} / 적용={a.get('priority','없음')}"
        print(f"   {L:5} 적용값 threshold={thr}, timeout={to}, priority={pr}{flag}")

def main():
    args = sys.argv[1:] or ["hrtt/demo"]
    files = []
    for a in args:
        if os.path.isdir(a): files += glob.glob(os.path.join(a, "*.hrtt")) + glob.glob(os.path.join(a, "*.html"))
        else: files.append(a)
    for f in sorted(files):
        check(f)

if __name__ == "__main__":
    main()
