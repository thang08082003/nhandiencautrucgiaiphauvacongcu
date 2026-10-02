"""Bước 4: đo tốc độ (batch=1). 10 lần khởi động + 100 lần đo (CT toàn khối: giảm còn 1 + 5 lần vì rất chậm).
python speed.py --name camus|cholec|ct
Latency = chỉ phần forward của mô hình (chưa gồm resize/hậu xử lý). Dice tính trên tập test.
"""
import argparse
import tempfile
import traceback

import matplotlib
import numpy as np
import pandas as pd
import torch

from common import CACHE, RESULTS, bench, mean_dice, start

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ort_session(model, shape):
    import onnxruntime as ort
    f = tempfile.mktemp(suffix=".onnx")
    torch.onnx.export(model.eval().float(), torch.randn(*shape, device=next(model.parameters()).device), f,
                      input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(f, providers=prov), prov[0]


def run_2d(name, dev):
    from lib2d import build, evaluate, load_cache, meta, predict
    names = meta(name)["names"]
    C, S = meta(name)["in_ch"], meta(name)["size"]
    model = build(name, pretrained=False).to(dev)
    model.load_state_dict(torch.load(CACHE / "ckpt" / f"{name}_best.pt"))
    model.eval()
    Xte, Yte = load_cache(name, "test")
    rows, errs = [], []
    for scale in (1.0, 0.75, 0.5):
        s = int(S * scale)
        x = torch.randn(1, C, s, s, device=dev)
        for prec in ("fp32", "fp16"):
            amp = prec == "fp16"

            def fn():
                with torch.no_grad(), torch.autocast("cuda", enabled=amp and dev == "cuda"):
                    model(x)
            r = bench(fn)
            r.update(backend="pytorch", precision=prec, input=f"{s}x{s}", scale=scale,
                     dice_mean_fg=mean_dice(evaluate(predict(model, Xte, scale, amp), Yte, names, hd=False)))
            rows.append(r)
            print(r, flush=True)
    try:  # ONNX Runtime ở 100%
        sess, prov = ort_session(model, (1, C, S, S))
        xn = np.random.randn(1, C, S, S).astype(np.float32)
        r = bench(lambda: sess.run(None, {"x": xn}))
        r["vram_peak_mb"] = np.nan  # torch không đo được VRAM của ORT
        from lib2d import prep
        pred = np.concatenate([sess.run(None, {"x": prep(torch.from_numpy(Xte[i:i + 1])).numpy()})[0].argmax(1).astype(np.uint8)
                               for i in range(len(Xte))])
        r.update(backend=f"onnxruntime/{prov}", precision="fp32", input=f"{S}x{S}", scale=1.0,
                 dice_mean_fg=mean_dice(evaluate(pred, Yte, names, hd=False)))
        rows.append(r)
    except Exception:
        errs.append("ONNX/ORT: " + traceback.format_exc())
    return rows, errs


def run_ct(dev):
    from monai.inferers import sliding_window_inference
    from train_ct import CT, PATCH, NCLS, build_model, load, split_ids
    model = build_model().to(dev)
    model.load_state_dict(torch.load(CT / "ckpt" / "ct_best.pt"))
    model.eval()
    rows, errs = [], []
    x = torch.randn(1, 1, *PATCH, device=dev)
    for prec in ("fp32", "fp16"):
        amp = prec == "fp16"

        def fn():
            with torch.no_grad(), torch.autocast("cuda", enabled=amp and dev == "cuda"):
                model(x)
        rows.append(dict(bench(fn), backend="pytorch", precision=prec, input=f"patch {PATCH}", scope="1 patch"))
    ids = split_ids("test")
    cid = sorted(ids, key=lambda c: np.load(CT / f"{c}_lab.npy", mmap_mode="r").size)[len(ids) // 2]  # ca cỡ trung vị
    vol = torch.from_numpy(load(cid)[0].astype(np.float32))[None, None].to(dev)
    for prec in ("fp32", "fp16"):
        amp = prec == "fp16"

        def fn():
            with torch.no_grad(), torch.autocast("cuda", enabled=amp and dev == "cuda"):
                sliding_window_inference(vol, PATCH, 4, model, overlap=0.25)
        rows.append(dict(bench(fn, warm=1, n=5), backend="pytorch", precision=prec,
                         input=f"{cid} {tuple(vol.shape[2:])}", scope="cả khối (sliding window, overlap .25)"))
    try:
        sess, prov = ort_session(model, (1, 1, *PATCH))
        xn = np.random.randn(1, 1, *PATCH).astype(np.float32)
        rows.append(dict(bench(lambda: sess.run(None, {"x": xn})), backend=f"onnxruntime/{prov}", precision="fp32",
                         input=f"patch {PATCH}", scope="1 patch", vram_peak_mb=np.nan))
    except Exception:
        errs.append("ONNX/ORT: " + traceback.format_exc())
    for r in rows:
        print(r, flush=True)
    return rows, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=["camus", "cholec", "ct"])
    a = ap.parse_args()
    start()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from lib2d import RDIR
    out = RESULTS / ("ct" if a.name == "ct" else RDIR[a.name])
    rows, errs = run_ct(dev) if a.name == "ct" else run_2d(a.name, dev)
    df = pd.DataFrame(rows)
    df["gpu"] = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    df.to_csv(out / "speed_benchmark.csv", index=False)
    if errs:
        (out / "speed_errors.txt").write_text("\n".join(errs))
        print("LỖI (đã ghi speed_errors.txt):", *errs)
    if a.name != "ct":
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for _, r in df.iterrows():
            ax.scatter(r.fps, r.dice_mean_fg)
            ax.annotate(f"{r.backend.split('/')[0]} {r.precision} {r.input}", (r.fps, r.dice_mean_fg), fontsize=6)
        ax.set(xlabel="FPS (batch=1, chỉ forward)", ylabel="Dice trung bình (test)", title=f"{a.name}: Dice vs FPS")
        fig.tight_layout()
        fig.savefig(out / "dice_vs_fps.png", dpi=120)


if __name__ == "__main__":
    main()
