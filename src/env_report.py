"""Bước 0: ghi results/env_report.md và results/requirements_frozen.txt."""
import platform
import shutil
import subprocess
import sys

from common import RESULTS, ROOT, start


def run(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main():
    start()
    RESULTS.mkdir(parents=True, exist_ok=True)
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        x = torch.randn(1024, 1024, device="cuda")
        ok = bool(torch.isfinite(x @ x).all())
    d = shutil.disk_usage(ROOT)
    (RESULTS / "env_report.md").write_text("\n".join([
        "# Môi trường", f"- OS: {platform.platform()}", f"- Python: {sys.version.split()[0]}",
        f"- RAM: {run('free -h | head -2') or run('sysctl -n hw.memsize')}",
        f"- Đĩa trống ({ROOT}): {d.free / 2**30:.0f} GB / {d.total / 2**30:.0f} GB",
        f"- PyTorch: {torch.__version__}, CUDA build: {torch.version.cuda}, GPU thấy được: {cuda}",
        f"- Kiểm tra nhanh GPU (matmul 1024x1024): {'OK' if cuda and ok else 'KHÔNG CHẠY ĐƯỢC / không có GPU'}",
        "", "```", run("nvidia-smi") or "không có nvidia-smi", "```", ""]))
    (RESULTS / "requirements_frozen.txt").write_text(run(f"{sys.executable} -m pip freeze"))
    print((RESULTS / "env_report.md").read_text())


if __name__ == "__main__":
    main()
