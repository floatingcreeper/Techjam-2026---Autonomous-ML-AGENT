"""FIXED node runner.

Assembles the six blocks from a snapshot dir, trains, evaluates on valid (and optionally
an extra split for the final submission), and writes metrics.json + score arrays. It never
imports anything from agent/roles or agent/orchestrator -- only the fixed harness + datced.

Run as a module so the repo root is on sys.path:
    python -m pipeline.run_node --blocks <dir> --out <dir> [--cfg cfg.json]
                                [--cache runs/_cache] [--extra-split test]
"""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np

from evaluate import evaluate                 # fixed harness
from agent.datced import load_bundle
from pipeline.contracts import Cfg


def _load_block(blocks_dir: str, name: str):
    p = Path(blocks_dir) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_node_block_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, help="dir with features/model/loss/train/infer/ensemble.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--cache", default="runs/_cache")
    ap.add_argument("--extra-split", default=None, help="also score this split (e.g. test) for the final submission")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Cfg.from_json(a.cfg) if a.cfg else Cfg()
    t0 = time.time()

    bundle = load_bundle(a.cache)
    F = _load_block(a.blocks, "features")
    M = _load_block(a.blocks, "model")
    L = _load_block(a.blocks, "loss")
    T = _load_block(a.blocks, "train")
    I = _load_block(a.blocks, "infer")

    feats = F.build_features(bundle, cfg)
    model = M.build_model(feats.meta, cfg)
    lossfn = L.build_loss(cfg)
    model = T.train(model, lossfn, feats, bundle, cfg)

    sv = np.asarray(I.infer(model, feats, "valid"))
    mv = evaluate(feats.users["valid"], feats.y["valid"], sv)
    np.save(out / "val_scores.npy", sv.astype(np.float32))

    res = {
        "GAUC": round(float(mv["GAUC"]), 6),
        "nDCG@5": round(float(mv["nDCG@5"]), 6),
        "primary_valid": round(float(mv["primary"]), 6),
        "primary_unbiased": None,
        "best_epoch": int(getattr(model, "_best_epoch", 0)),
        "seconds": round(time.time() - t0, 1),
    }

    if a.extra_split and a.extra_split != "valid":
        ss = np.asarray(I.infer(model, feats, a.extra_split))
        np.save(out / f"{a.extra_split}_scores.npy", ss.astype(np.float32))

    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
