from pathlib import Path

def get_custom_metadata(info, audio):
    audio_path = Path(info["path"])
    prompt = audio_path.parent.stem.replace("_", " ")

    return {"prompt": prompt}
