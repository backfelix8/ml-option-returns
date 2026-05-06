#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_shap_ladder.py

Compute SHAP (Jakob-style beeswarm + shapley_summary.xlsx) separately for each
feature_fraction ladder level L0..L4, where ladder levels are stored as the
run-index j in filenames:

    {model_name}_model{i}_{j}.pkl

i = rolling window index (0..N-1)
j = ladder level index (0..4)  -> L0..L4

This script does NOT retrain models and does NOT perturb y/Returns.
It reuses evaluate.preprocess_for_model(full_dataset=True) and
CombinedModel.shapley_values(...) for consistent deliverables.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import argparse
import shutil
from pathlib import Path

def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))

def _prepare_level_models(
    src_folder: Path,
    tmp_folder: Path,
    model_name: str,
    N: int,
    level: int,
    also_sidecars: bool = True,
) -> None:
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # sidecar patterns observed in your folder:
    #   .txt
    #   _meta.json
    sidecars = [".txt", "_meta.json"] if also_sidecars else []

    for i in range(N):
        base_src = src_folder / f"{model_name}_model{i}_{level}"
        base_dst = tmp_folder / f"{model_name}_model{i}_0"

        # required
        src_pkl = base_src.with_suffix(".pkl")
        dst_pkl = base_dst.with_suffix(".pkl")
        if not src_pkl.exists():
            raise FileNotFoundError(f"Missing model file: {src_pkl}")
        _link_or_copy(src_pkl, dst_pkl)

        # optional sidecars
        for suf in sidecars:
            src_sc = Path(str(base_src) + suf)
            if src_sc.exists():
                dst_sc = Path(str(base_dst) + suf)
                _link_or_copy(src_sc, dst_sc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_folder", type=str, required=True,
                    help="Folder containing ladder models like <name>_model{i}_{j}.pkl")
    ap.add_argument("--model_name", type=str, default="gbr_ffladder",
                    help="Model filename prefix (e.g., gbr_ffladder)")
    ap.add_argument("--model_type", type=str, default="rf",
                    help="Jakob model type string for CombinedModel (tree models use 'rf')")
    ap.add_argument("--N", type=int, default=3,
                    help="Number of rolling windows i (default 3)")
    ap.add_argument("--levels", type=str, default="0,1,2,3,4",
                    help="Comma-separated ladder levels, default 0,1,2,3,4")
    ap.add_argument("--out_root", type=str, default="./shap_ladder",
                    help="Output root folder")
    ap.add_argument("--selected_data_path", type=str, default="",
                    help="Optional override for evaluate.SELECTED_DATA_PATH")
    ap.add_argument("--no_sidecars", action="store_true",
                    help="Do not link/copy .txt and _meta.json sidecars")
    args = ap.parse_args()

    src_folder = Path(args.src_folder).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Make local imports robust (script in same repo as evaluate.py / CombinedModel.py)
    repo_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_dir))

    import evaluate  # noqa
    from CombinedModel import CombinedModel  # noqa

    if args.selected_data_path:
        evaluate.SELECTED_DATA_PATH = args.selected_data_path

    # Load full 1..12 months (needed by shapley_values background construction)
    X_months, _y_months = evaluate.preprocess_for_model(full_dataset=True)
    print(f"[data] months loaded = 1..12, n_months={len(X_months)}")
    print(f"[data] rows per month = {[len(df) for df in X_months]}")
    print(f"[data] n_features = {X_months[0].shape[1]}")

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip() != ""]
    print(f"[config] model_name={args.model_name} model_type={args.model_type} N={args.N} levels={levels}")
    print("[note] This computes SHAP on held-out months as defined inside CombinedModel.shapley_values().")

    for L in levels:
        out_L = out_root / f"L{L}"
        out_L.mkdir(parents=True, exist_ok=True)

        tmp_models = out_L / "tmp_models"
        figs = out_L / "figures"
        figs.mkdir(parents=True, exist_ok=True)

        print(f"\n===== [SHAP] Level L{L} =====")
        print(f"[paths] src_folder={src_folder}")
        print(f"[paths] out={out_L}")

        # Prepare temp model folder with j=0 only
        _prepare_level_models(
            src_folder=src_folder,
            tmp_folder=tmp_models,
            model_name=args.model_name,
            N=args.N,
            level=L,
            also_sidecars=(not args.no_sidecars),
        )
        print(f"[models] prepared temp folder: {tmp_models}")

        # Instantiate CombinedModel with amount=1 (only j=0)
        models = CombinedModel(
            str(tmp_models),
            1,                      # amount=1
            args.N,                 # N rolling windows
            [args.model_type],      # model_types
            [args.model_name],      # model_names (filename prefix)
            [1.0],                  # model_weights
            evaluate.IMPORTANT_COLUMNS,
            [False],                # normalize
        )

        # Ensure shapley_summary.xlsx is written into out_L
        cwd0 = Path.cwd()
        try:
            os.chdir(out_L)
            models.shapley_values(X_months, X_months, fig_dir=str(figs))
        finally:
            os.chdir(cwd0)

        # Safety rename (optional): keep default shapley_summary.xlsx but also give explicit name
        src_xlsx = out_L / "shapley_summary.xlsx"
        if src_xlsx.exists():
            dst_xlsx = out_L / f"shapley_summary_L{L}.xlsx"
            shutil.copy2(src_xlsx, dst_xlsx)
            print(f"[ok] wrote: {dst_xlsx}")
        else:
            print("[warn] shapley_summary.xlsx not found; please check CombinedModel.shapley_values output.")

        print(f"[ok] figures in: {figs}")

    print("\n[done] SHAP ladder computation complete.")

if __name__ == "__main__":
    main()

