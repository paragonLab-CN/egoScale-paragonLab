import torch
import transformers

model_name = "/root/autodl-tmp/le-wm-main/models/Cosmos-Reason2-2B"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=dtype,
    device_map="auto",
    attn_implementation="sdpa",
)

processor = transformers.AutoProcessor.from_pretrained(model_name)

image_path = "/root/autodl-tmp/le-wm-main/test.jpg"

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a robotics visual encoder."}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                #"image": f"file://{image_path}",
                "image": image_path,
            },
            {
                "type": "text",
                "text": "Encode this robot scene.",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

last_hidden = outputs.hidden_states[-1]

print("last_hidden shape:", last_hidden.shape)

pooled_feat = last_hidden.mean(dim=1)

print("pooled feature shape:", pooled_feat.shape)