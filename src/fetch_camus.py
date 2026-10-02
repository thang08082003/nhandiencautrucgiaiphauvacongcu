"""Tải CAMUS từ API chính thức (CREATIS, Girder) theo từng file, song song, tự thử lại, bỏ qua file đã có.
Chỉ lấy file cần cho đề tài: ảnh + nhãn ED/ES, Info_*.cfg, file chia tập, giấy phép. Bỏ file *_sequence* (nặng, không dùng).
Giấy phép: CC BY-NC-SA 4.0, chỉ dùng cho nghiên cứu phi thương mại; trích dẫn bắt buộc (xem LICENSE_TERMS.md tải kèm).
python fetch_camus.py [--out /content/data/camus] [--limit N]
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common import DATA, start

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
COLL = "6373703d73e9f0047faa1bc8"
KEEP = re.compile(r"(_(ED|ES)(_gt)?\.nii\.gz|^Info_\w+\.cfg|^MANDATORY_CITATION\.md|\.txt|\.md)$")
S = requests.Session()


def get(path, **params):
    for i in range(8):
        try:
            r = S.get(f"{API}/{path}", params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == 7:
                raise
            time.sleep(3 * (i + 1))


def fetch(job):
    item_id, size, dest = job
    if dest.exists() and dest.stat().st_size == size:
        return "có sẵn"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for i in range(10):
        try:
            have = part.stat().st_size if part.exists() else 0
            h = {"Range": f"bytes={have}-"} if have else {}
            with S.get(f"{API}/item/{item_id}/download", headers=h, stream=True, timeout=60) as r:
                if r.status_code == 200:  # server bỏ qua Range -> tải lại từ đầu
                    have = 0
                r.raise_for_status()
                with open(part, "ab" if have else "wb") as f:
                    for ch in r.iter_content(1 << 16):
                        f.write(ch)
            if part.stat().st_size == size:
                part.rename(dest)
                return "ok"
            raise IOError(f"sai dung lượng {part.stat().st_size}/{size}")
        except Exception as e:
            time.sleep(3 * (i + 1))
    return f"LỖI {dest.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "camus"))
    ap.add_argument("--limit", type=int, default=0, help="chỉ tải N bệnh nhân đầu (để thử)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    start()
    out = Path(a.out) / "CAMUS_public"
    jobs = []
    for top in get("folder", parentType="collection", parentId=COLL, limit=0):
        if top["name"] == "jupyter":
            continue
        folders = [(top["name"], top["_id"])]
        if top["name"] == "database_nifti":
            pats = get("folder", parentType="folder", parentId=top["_id"], limit=0)
            pats = sorted(pats, key=lambda x: x["name"])[: a.limit or None]
            folders = [(f"database_nifti/{p['name']}", p["_id"]) for p in pats]
        elif a.limit:
            pass
        for rel, fid in folders:
            for it in get("item", folderId=fid, limit=0):
                if KEEP.search(it["name"]):
                    jobs.append((it["_id"], it["size"], out / rel / it["name"]))
    mb = sum(j[1] for j in jobs) / 2**20
    print(f"{len(jobs)} file, {mb:.0f} MB. Nguồn: {API} (CAMUS chính thức), CC BY-NC-SA 4.0 phi thương mại.", flush=True)
    done = {"ok": 0, "có sẵn": 0}
    errs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(fetch, jobs), 1):
            done[r] = done.get(r, 0) + 1
            if r.startswith("LỖI"):
                errs.append(r)
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("Kết quả:", {k: v for k, v in done.items() if not k.startswith("LỖI")}, "| lỗi:", len(errs))
    if errs:
        print(*errs[:20], sep="\n")
        raise SystemExit("Còn file lỗi: chạy lại lệnh này để tải tiếp (file đã tải sẽ được bỏ qua).")


if __name__ == "__main__":
    main()
