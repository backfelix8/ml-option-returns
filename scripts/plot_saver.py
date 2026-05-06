from __future__ import annotations

from pathlib import Path
import re


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._-") or "figure"


def save_current_figure(fig_dir, name: str, ext: str = "png", dpi: int = 300, close: bool = True):
    from matplotlib import pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_filename(name)
    ext = ext.lstrip(".")
    out_path = fig_dir / f"{safe_name}.{ext}"

    fig = plt.gcf()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_path
