import os
import glob
import random
import argparse

import torch
from torch.optim import AdamW
from tqdm import tqdm

from lewm_cosmos_adapter import CosmosToLeWMAdapter, alignment_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/root/autodl-tmp/le-wm-main/alignment_cache",
    )

    parser.add_argument(
        "--train_cache_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--val_cache_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/root/autodl-tmp/le-wm-main/cosmos_to_lewm_adapter.pt",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--cos_weight", type=float, default=0.1)

    return parser.parse_args()

def run_one_epoch(adapter, paths, device, cos_weight, optimizer=None, train=True):
    if train:
        adapter.train()
        random.shuffle(paths)
        desc = "train"
    else:
        adapter.eval()
        desc = "val"

    total_loss = 0.0
    total_mse = 0.0
    total_cos = 0.0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for path in tqdm(paths, desc=desc):
            data = torch.load(path, map_location="cpu")

            cosmos_feats = data["cosmos"].float().to(device)
            lewm_emb = data["lewm"].float().to(device)

            assert cosmos_feats.ndim == 3, cosmos_feats.shape
            assert lewm_emb.ndim == 3, lewm_emb.shape

            assert cosmos_feats.shape[-1] == 2048, cosmos_feats.shape
            assert lewm_emb.shape[-1] == 192, lewm_emb.shape

            assert cosmos_feats.shape[0] == lewm_emb.shape[0]
            assert cosmos_feats.shape[1] == lewm_emb.shape[1]

            pred = adapter(cosmos_feats)

            assert pred.shape == lewm_emb.shape, f"pred {pred.shape}, target {lewm_emb.shape}"

            loss, logs = alignment_loss(
                pred=pred,
                target=lewm_emb,
                cos_weight=cos_weight,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            total_loss += logs["total_loss"]
            total_mse += logs["mse_loss"]
            total_cos += logs["cos_loss"]

    n = len(paths)

    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "cos": total_cos / n,
    }


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("cache_dir:", args.cache_dir)

    train_cache_dir = args.train_cache_dir or os.path.join(args.cache_dir, "train")
    val_cache_dir = args.val_cache_dir or os.path.join(args.cache_dir, "val")

    train_paths = sorted(glob.glob(os.path.join(train_cache_dir, "*.pt")))
    val_paths = sorted(glob.glob(os.path.join(val_cache_dir, "*.pt")))

    assert len(train_paths) > 0, f"没有找到 train cache 文件: {train_cache_dir}"
    assert len(val_paths) > 0, f"没有找到 val cache 文件: {val_cache_dir}"

    print("train_cache_dir:", train_cache_dir)
    print("val_cache_dir:", val_cache_dir)
    print("train cache file number:", len(train_paths))
    print("val cache file number:", len(val_paths))

    adapter = CosmosToLeWMAdapter(
        in_dim=2048,
        hidden_dim=args.hidden_dim,
        out_dim=192,
    ).to(device)

    optimizer = AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        train_logs = run_one_epoch(
            adapter=adapter,
            paths=train_paths,
            device=device,
            cos_weight=args.cos_weight,
            optimizer=optimizer,
            train=True,
        )

        val_logs = run_one_epoch(
            adapter=adapter,
            paths=val_paths,
            device=device,
            cos_weight=args.cos_weight,
            optimizer=None,
            train=False,
        )

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_logs['loss']:.6f} | "
            f"train_mse={train_logs['mse']:.6f} | "
            f"train_cos={train_logs['cos']:.6f} | "
            f"val_loss={val_logs['loss']:.6f} | "
            f"val_mse={val_logs['mse']:.6f} | "
            f"val_cos={val_logs['cos']:.6f}"
        )

        # 每轮都保存 latest
        latest_path = args.save_path.replace(".pt", "_latest.pt")
        torch.save(adapter.state_dict(), latest_path)

        # 用 val_loss 保存 best，而不是 train_loss
        if val_logs["loss"] < best_val_loss:
            best_val_loss = val_logs["loss"]
            torch.save(adapter.state_dict(), args.save_path)
            print("saved best adapter:", args.save_path)

    print("training finished")
    print("best val loss:", best_val_loss)
    print("best adapter path:", args.save_path)


if __name__ == "__main__":
    main()