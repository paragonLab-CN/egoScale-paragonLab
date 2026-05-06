import sys
from modelscope import snapshot_download

if __name__ == "__main__":
    # 强制实时输出，避免缓冲导致看不到打印
    print("脚本开始执行...", flush=True)

    try:
        print("正在准备下载模型，请稍候...", flush=True)
        model_dir = snapshot_download(
            model_id='Qwen/Qwen3.5-4B-Base',
            cache_dir='./'
        )
        print(f"模型下载完成，位置：{model_dir}", flush=True)
    except Exception as e:
        print(f"下载失败，错误信息：{e}", flush=True)
        sys.exit(1)