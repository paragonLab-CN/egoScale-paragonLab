import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    try:
        # 1. 脚本启动
        print("1. 脚本开始运行...", flush=True)

        # 模型路径
        model_path = "/root/autodl-tmp/le-wm-main/Qwen/Qwen3___5-4B-Base"
        print(f"2. 尝试从 {model_path} 加载模型...", flush=True)

        # 2. 加载 Tokenizer
        print("3. 加载 Tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("   Tokenizer 加载完成。", flush=True)

        # 3. 加载模型
        print(f"4. 开始加载模型 (torch.bfloat16, device_map='auto')...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("   模型加载完成。", flush=True)

        # 4. 测试推理
        text = "robot arm pick up the block"
        print(f"5. 准备处理文本: '{text}'", flush=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        print("6. 运行模型进行推理...", flush=True)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # 5. 输出结果
        last_hidden_state = outputs.hidden_states[-1]
        print(f"7. 模型运行完成! last_hidden_state 的形状是: {last_hidden_state.shape}", flush=True)

    except Exception as e:
        print(f"\n❌ 脚本发生错误: {e}", flush=True)
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
        sys.exit(1)