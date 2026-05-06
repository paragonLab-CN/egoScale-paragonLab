import os
import tempfile

import torch
import transformers
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def tensor_to_pil_image(x):
    """
    把 LeWM 里的 tensor 图片转成 PIL 图片。

    输入:
        x: [3, H, W]

    输出:
        PIL.Image
    """

    x = x.detach().cpu()

    # 如果是 float，默认希望范围是 [0, 1]
    if x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        x = x.float()

        #print("tensor image min/max before clamp:", x.min().item(), x.max().item())

        # 先保守处理，防止越界导致 PIL 转换报错
        x = x.clamp(0, 1)

    image = to_pil_image(x)
    image = image.convert("RGB")

    return image


class CosmosFeatureExtractor:
    """
    Cosmos Reason2 特征提取器。

    作用：
        1. 加载 Cosmos Reason2 模型
        2. 输入单张图片路径，输出 [1, 2048]
        3. 输入多张图片路径，输出 [N, 2048]
        4. 输入 LeWM pixels [B, T, 3, H, W]，输出 [B, T, 2048]
    """

    def __init__(
        self,
        model_name="/root/autodl-tmp/le-wm-main/models/Cosmos-Reason2-2B",
        device=None,
    ):
        """
        model_name:
            你的 Cosmos Reason2 本地模型路径。

        device:
            一般不用传。
            如果有 cuda，会自动用 cuda。
        """

        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("Loading Cosmos Reason2 model from:", self.model_name)

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        print("Using dtype:", dtype)

        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )

        self.processor = transformers.AutoProcessor.from_pretrained(self.model_name)

        self.model.eval()

        # 冻结 Cosmos，不训练 Cosmos，只提特征
        for p in self.model.parameters():
            p.requires_grad = False

        # 尽量和你原始脚本保持一致：inputs.to(model.device)
        # 如果 model.device 不存在，就用第一个参数所在 device
        if hasattr(self.model, "device"):
            self.input_device = self.model.device
        else:
            self.input_device = next(self.model.parameters()).device

        print("Cosmos input device:", self.input_device)

    def _build_messages(self, image_path):
        """
        构造和你原始脚本一样的 messages。
        """

        assert isinstance(image_path, str), "image_path 必须是字符串"
        assert not image_path.startswith("file://"), "不要加 file://，直接用 /root/xxx/test.jpg"
        assert os.path.exists(image_path), f"图片不存在: {image_path}"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a robotics visual encoder.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": "Encode this robot scene.",
                    },
                ],
            },
        ]

        return messages

    @torch.no_grad()
    def extract_one_image(self, image_path, return_last_hidden=False):
        """
        提取单张图片的 Cosmos 特征。

        输入:
            image_path: 图片路径，例如 "/root/autodl-tmp/le-wm-main/test.jpg"

        输出:
            默认返回 pooled_feat: [1, 2048]

            如果 return_last_hidden=True，
            返回:
                last_hidden: [1, 1183, 2048]
                pooled_feat: [1, 2048]
        """

        messages = self._build_messages(image_path)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.input_device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]

        # last_hidden: [1, 1183, 2048]
        # pooled_feat: [1, 2048]
        pooled_feat = last_hidden.mean(dim=1)

        #print("last_hidden shape:", last_hidden.shape)
        #print("pooled feature shape:", pooled_feat.shape)

        assert pooled_feat.ndim == 2, pooled_feat.shape
        assert pooled_feat.shape[0] == 1, pooled_feat.shape
        assert pooled_feat.shape[1] == 2048, pooled_feat.shape

        if return_last_hidden:
            return last_hidden, pooled_feat
        else:
            return pooled_feat

    @torch.no_grad()
    def extract_image_paths(self, image_paths):
        """
        提取多张图片路径的 Cosmos 特征。

        输入:
            image_paths: list[str]

        输出:
            feats: [N, 2048]
        """

        all_feats = []

        for i, image_path in enumerate(image_paths):
            #print(f"Extracting image {i + 1}/{len(image_paths)}:", image_path)

            feat = self.extract_one_image(image_path)

            # feat: [1, 2048]
            all_feats.append(feat.detach().cpu())

        feats = torch.cat(all_feats, dim=0)

        # feats: [N, 2048]
        #print("all image path feats shape:", feats.shape)

        return feats

    @torch.no_grad()
    def extract_tensor_images(self, images):
        """
        提取 tensor 图片的 Cosmos 特征。

        输入:
            images: [N, 3, H, W]

        输出:
            feats: [N, 2048]

        注意：
            因为你当前跑通的 Cosmos 输入方式是 image_path，
            所以这里会临时把 tensor 保存成 png，再送进 Cosmos。
            这样最稳，但速度会慢一点。
        """

        assert images.ndim == 4, images.shape
        assert images.shape[1] == 3, images.shape

        N = images.shape[0]

        all_feats = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(N):
                #print(f"Extracting tensor image {i + 1}/{N}")

                image_tensor = images[i]
                pil_image = tensor_to_pil_image(image_tensor)

                tmp_path = os.path.join(tmpdir, f"tmp_image_{i:06d}.png")
                pil_image.save(tmp_path)

                feat = self.extract_one_image(tmp_path)

                # feat: [1, 2048]
                all_feats.append(feat.detach().cpu())

        feats = torch.cat(all_feats, dim=0)

        # feats: [N, 2048]
        #print("tensor image feats shape:", feats.shape)

        return feats

    @torch.no_grad()
    def extract_lewm_pixels(self, pixels):
        """
        专门给 LeWM 用。

        输入:
            pixels: [B, T, 3, H, W]

        输出:
            cosmos_feats: [B, T, 2048]
        """

        assert pixels.ndim == 5, pixels.shape

        B, T, C, H, W = pixels.shape

        assert C == 3, pixels.shape

        #print("LeWM pixels shape:", pixels.shape)

        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        flat_pixels = pixels.reshape(B * T, C, H, W)

        #print("flat pixels shape:", flat_pixels.shape)

        # [B*T, 3, H, W] -> [B*T, 2048]
        flat_feats = self.extract_tensor_images(flat_pixels)

        # [B*T, 2048] -> [B, T, 2048]
        cosmos_feats = flat_feats.reshape(B, T, 2048)

        #print("cosmos feats for LeWM shape:", cosmos_feats.shape)

        return cosmos_feats