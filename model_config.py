"""Resolve model weights from studio_config.json, independently of the process cwd."""
import json
from pathlib import Path


def load_model_paths():
    config_path = Path(__file__).resolve().with_name("studio_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    directory = config.get("model_directory") if isinstance(config, dict) else None
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError(f"{config_path}: model_directory must be a non-empty directory path")
    root = Path(directory.strip()).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    filenames = {
        "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
        "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "spatial_upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    }
    return {name: str(root / filename) for name, filename in filenames.items()}
def load_ltx_directory():
    config_path = Path(__file__).resolve().with_name("studio_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    directory = config.get("ltx_directory") if isinstance(config, dict) else None
    if not isinstance(directory, str) or not directory.strip():
             raise ValueError(f"{config_path}: ltx_directory must be a non-empty directory path")
    return directory.strip()