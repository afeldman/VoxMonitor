import os
import yaml
import torch
from voxmonitor.utils import dynamic_import
from voxmonitor.model import MultiTaskMobileNet


def main(cfg_path: str = "config/config.yaml", ckpt_path: str = "checkpoints/best.pt", out_name: str = "model.onnx"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    onnx_cfg = cfg.get("onnx", {})

    os.makedirs(onnx_cfg.get("export_dir", "exports/onnx"), exist_ok=True)
    out_path = os.path.join(onnx_cfg.get("export_dir", "exports/onnx"), out_name)

    # Load checkpoint meta (num_classes, label_mappings)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt.get("num_classes")
    state_dict = ckpt.get("model_state")

    # Try dynamic exporter first
    exporter_path = onnx_cfg.get("exporter_path", "")
    if exporter_path:
        try:
            exporter, _, _ = dynamic_import(exporter_path)
            exporter(ckpt_path=ckpt_path, out_path=out_path, cfg=cfg)
            print(f"Exported via custom exporter to {out_path}")
            return
        except Exception as e:
            print(f"[WARN] Custom exporter failed: {e}. Fallback to torch.onnx.export.")

    # Fallback: rebuild model and export
    model = MultiTaskMobileNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input: (1,3,H,W). Use typical Mel dims; adjust if needed.
    dummy = torch.randn(1, 3, 128, 256)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["mel"],
        output_names=[f"logits_{i}" for i in range(len(num_classes))],
        opset_version=onnx_cfg.get("opset", 17),
        dynamic_axes={"mel": {2: "n_mels", 3: "time"}},
    )
    print(f"Exported ONNX to {out_path}")


if __name__ == "__main__":
    raise NotImplementedError("ONNX-Export erfolgt über aptt-Exporter. Bitte aptt verwenden.")
