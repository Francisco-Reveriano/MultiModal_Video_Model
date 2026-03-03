import hashlib
import json
import os
import re

from src.captioning.gemini import caption_all


_FLIP_WORD_MAP = {
    r"\bleft\b": "right",
    r"\bright\b": "left",
}

_STYLE_VARIANTS = [
    "Cinematic composition, natural lighting, realistic motion detail.",
    "Photorealistic detail, clean motion continuity, dynamic camera movement.",
    "High-fidelity visual detail, stable framing, and smooth subject motion.",
]


def _apply_flip_caption_rules(caption: str) -> str:
    transformed = caption
    for pattern, replacement in _FLIP_WORD_MAP.items():
        transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
    return transformed


def _pick_variant_index(key: str, num_variants: int) -> int:
    if num_variants <= 1:
        return 0
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_variants


def _augment_caption_text(
    caption: str,
    filename: str,
    augmentation: str,
    is_flipped: bool,
    rule_variants: int = 1,
) -> str:
    result = caption.strip()

    if is_flipped:
        result = _apply_flip_caption_rules(result)

    if "temporal_crop" in augmentation:
        result = f"{result} Short action-focused crop from the original sequence."

    variant_count = min(max(rule_variants, 1), len(_STYLE_VARIANTS))
    variant_idx = _pick_variant_index(filename, variant_count)
    result = f"{result} {_STYLE_VARIANTS[variant_idx]}"
    return result


def caption_augmented_videos(
    video_records: list[dict],
    video_dir: str,
    dataset_dir: str,
    trigger_token: str,
    context: str,
    api_key: str,
    per_video_context: dict | None = None,
    delay: float = 1.0,
    rule_variants: int = 1,
) -> dict[str, str]:
    """
    Caption augmented videos with Gemini, then apply rule-based caption augmentation.
    Writes training manifests compatible with downstream training scripts.
    """
    os.makedirs(dataset_dir, exist_ok=True)

    raw_captions = caption_all(
        video_dir=video_dir,
        dataset_dir=dataset_dir,
        trigger_token=trigger_token,
        context=context,
        api_key=api_key,
        per_video_context=per_video_context or {},
        delay=delay,
    )

    by_filename = {record["filename"]: record for record in video_records}
    augmented_captions: dict[str, str] = {}
    for filename, caption in raw_captions.items():
        if caption.startswith("[FAILED]"):
            augmented_captions[filename] = caption
            continue

        record = by_filename.get(filename, {})
        augmentation = record.get("augmentation", "base")
        is_flipped = bool(record.get("is_flipped", False))
        augmented_captions[filename] = _augment_caption_text(
            caption=caption,
            filename=filename,
            augmentation=augmentation,
            is_flipped=is_flipped,
            rule_variants=rule_variants,
        )

    # Write augmented captions for traceability
    with open(os.path.join(dataset_dir, "captions_augmented.json"), "w") as f:
        json.dump(augmented_captions, f, indent=2)

    # Keep compatibility with existing training pipeline files
    with open(os.path.join(dataset_dir, "captions.json"), "w") as f:
        json.dump(augmented_captions, f, indent=2)

    successful = {k: v for k, v in augmented_captions.items() if not v.startswith("[FAILED]")}

    with open(os.path.join(dataset_dir, "videos.txt"), "w") as f:
        for filename in successful:
            f.write(f"{video_dir}/{filename}\n")

    with open(os.path.join(dataset_dir, "prompts.txt"), "w") as f:
        for caption in successful.values():
            f.write(f"{caption}\n")

    print(f"{'=' * 60}")
    print(f"Final augmented captions: {len(successful)}/{len(augmented_captions)}")

    return augmented_captions
