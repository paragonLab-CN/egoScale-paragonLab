import os
import torch

from cosmos_feature_extractor import CosmosFeatureExtractor


_COSMOS_EXTRACTOR = None
_CACHE_STEP_BY_STAGE = {
    "train": 0,
    "val": 0,
}


def _is_rank0():
    """
    防止多卡训练时每张卡都保存一份。
    单卡时默认就是 rank0。
    """

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    return rank == 0 and local_rank == 0


def maybe_dump_alignment_cache(raw_pixels, lewm_emb, stage=None):
    """
    在 LeWM 的 forward 里调用这个函数。

    输入:
        raw_pixels: [B, T, 3, H, W]
        lewm_emb:   [B, T, 192]

    保存:
        train 阶段:
            cache_dir/train/align_train_xxxxxx.pt

        val 阶段:
            cache_dir/val/align_val_xxxxxx.pt
    """

    global _COSMOS_EXTRACTOR
    global _CACHE_STEP_BY_STAGE

    enabled = os.environ.get("DUMP_ALIGNMENT_CACHE", "0") == "1"

    if not enabled:
        return

    if not _is_rank0():
        return
    
    if stage is None:
        return

    stage = str(stage).lower()

    # 兼容训练阶段名字
    if stage in ["training", "fit"]:
        stage = "train"

    # 兼容验证阶段名字
    if stage in ["valid", "validate", "validation"]:
        stage = "val"

    # 更保险：处理 runningstage.training / runningstage.validating 这类字符串
    if "train" in stage:
        stage = "train"
    elif "val" in stage or "valid" in stage:
        stage = "val"

    if stage not in ["train", "val"]:
        print(f"[Alignment Cache] skip unknown stage: {stage}")
        return

    max_batches = int(os.environ.get("ALIGNMENT_CACHE_MAX_BATCHES", "100000000"))
    cache_b = os.environ.get("ALIGNMENT_CACHE_B", "all")
    cache_dir = os.environ.get(
        "ALIGNMENT_CACHE_DIR",
        "/root/autodl-tmp/le-wm-main/alignment_cache",
    )
    cosmos_model_name = os.environ.get(
        "COSMOS_MODEL_NAME",
        "/root/autodl-tmp/le-wm-main/models/Cosmos-Reason2-2B",
    )

    step = _CACHE_STEP_BY_STAGE.get(stage, 0)

    # 注意：这里不要 raise SystemExit。
    # 否则 train 存够以后可能直接停止，val 还没来得及存。
    if step >= max_batches:
        return

    assert raw_pixels.ndim == 5, f"raw_pixels shape 应该是 [B,T,3,H,W]，但得到 {raw_pixels.shape}"
    assert lewm_emb.ndim == 3, f"lewm_emb shape 应该是 [B,T,192]，但得到 {lewm_emb.shape}"

    B, T, C, H, W = raw_pixels.shape

    assert C == 3, f"图片通道数应该是 3，但得到 {C}"
    assert lewm_emb.shape[0] == B, f"raw_pixels B={B}, lewm_emb B={lewm_emb.shape[0]}"
    assert lewm_emb.shape[1] == T, f"raw_pixels T={T}, lewm_emb T={lewm_emb.shape[1]}"
    assert lewm_emb.shape[2] == 192, f"lewm_emb 最后一维应该是 192，但得到 {lewm_emb.shape}"

    stage_cache_dir = os.path.join(cache_dir, stage)
    os.makedirs(stage_cache_dir, exist_ok=True)

    if cache_b.lower() in ["all", "full", "-1", "0"]:
        save_b = B
    else:
        save_b = int(cache_b)
        assert 1 <= save_b <= B, f"ALIGNMENT_CACHE_B={save_b} 不合法，当前 batch B={B}"

    # 全量缓存：默认保存当前 batch 的全部样本
    small_pixels = raw_pixels[:save_b].detach().cpu()
    small_lewm_emb = lewm_emb[:save_b].detach().cpu()

    #print("=" * 80)
    #print("[Alignment Cache] raw_pixels:", raw_pixels.shape)
    #print("[Alignment Cache] small_pixels:", small_pixels.shape)
    #print("[Alignment Cache] small_lewm_emb:", small_lewm_emb.shape)
    #print("[Alignment Cache] pixel min/max:", small_pixels.min().item(), small_pixels.max().item())

    if _COSMOS_EXTRACTOR is None:
        print("[Alignment Cache] 初始化 CosmosFeatureExtractor")
        _COSMOS_EXTRACTOR = CosmosFeatureExtractor(
            model_name=cosmos_model_name,
        )

    with torch.no_grad():
        cosmos_feats = _COSMOS_EXTRACTOR.extract_lewm_pixels(small_pixels)

    # cosmos_feats: [cache_b, T, 2048]
    assert cosmos_feats.ndim == 3, cosmos_feats.shape
    assert cosmos_feats.shape[0] == small_lewm_emb.shape[0], cosmos_feats.shape
    assert cosmos_feats.shape[1] == small_lewm_emb.shape[1], cosmos_feats.shape
    assert cosmos_feats.shape[2] == 2048, cosmos_feats.shape

    save_path = os.path.join(
        stage_cache_dir,
        f"align_{stage}_{step:06d}.pt",
    )

    torch.save(
        {
            "stage": stage,
            "cosmos": cosmos_feats.cpu().half(),
            "lewm": small_lewm_emb.cpu().float(),
        },
        save_path,
    )

    #print(
    #    f"[Alignment Cache] saved {stage} cache:",
    #    save_path,
    #    "cosmos:",
    #    tuple(cosmos_feats.shape),
    #    "lewm:",
    #    tuple(small_lewm_emb.shape),
    #)

    _CACHE_STEP_BY_STAGE[stage] = step + 1