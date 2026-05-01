import torch
from pathlib import Path

PROJECT_ROOT = Path(r"C:/face-recognition-fyp")

paths = [
    PROJECT_ROOT / "checkpoints" / "best_resnet_pretrained_stage2.pth",
    PROJECT_ROOT / "checkpoints" / "best_resnet_pretrained_stage1.pth",
    PROJECT_ROOT / "checkpoints" / "best_resnet_pretrained.pth",
    PROJECT_ROOT / "checkpoints" / "best_resnet_scratch.pth",
    PROJECT_ROOT / "outputs" / "checkpoints_improved" / "improved_best.pth",
    PROJECT_ROOT / "outputs" / "checkpoints" / "baseline_best.pth",
]

for path in paths:
    print("\n" + "=" * 80)
    print("FILE:", path)
    print("EXISTS:", path.exists())

    if not path.exists():
        continue

    try:
        ckpt = torch.load(path, map_location="cpu")
        print("TYPE:", type(ckpt))

        if isinstance(ckpt, dict):
            print("TOP-LEVEL KEYS:", list(ckpt.keys())[:20])

            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                state = ckpt["model_state_dict"]
                print("USING: model_state_dict")
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
                print("USING: state_dict")
            else:
                state = ckpt if all(isinstance(k, str) for k in ckpt.keys()) else None
                print("USING: raw dict" if state is not None else "NO DIRECT STATE DICT FOUND")

            if state is not None:
                keys = list(state.keys())[:20]
                print("FIRST PARAM KEYS:", keys)

                has_module_prefix = any(str(k).startswith("module.") for k in state.keys())
                print("HAS module. PREFIX:", has_module_prefix)

                for k, v in state.items():
                    if hasattr(v, "shape"):
                        print("FIRST TENSOR KEY:", k, "SHAPE:", tuple(v.shape))
                        break
        else:
            print("Checkpoint is not a dict, inspect manually.")

    except Exception as e:
        print("LOAD ERROR:", e)