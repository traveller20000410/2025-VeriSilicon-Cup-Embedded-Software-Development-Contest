import torch
import numpy as np
import json
from model import CNN

# 加载模型
device = torch.device("cpu")
model = CNN().to(device)
model.load_state_dict(
    torch.load("model/voice_detector_trained_956_971.pth", map_location=device, weights_only=True))
    # torch.load("model/speaker_classifier_trained_906_894_867.pth", map_location=device, weights_only=True))
model.eval()


def save_weights_to_bin(model, output_prefix="weights"):
    #导出权重为二进制文件(.bin)和元数据文件(.json)
    state_dict = model.state_dict()
    metadata = {}

    with open(f"{output_prefix}.bin", "wb") as bin_file:
        for key, tensor in state_dict.items():
            # 将张量转为numpy数组并写入二进制文件
            np_arr = tensor.detach().numpy().astype(np.float32)
            np_arr.tofile(bin_file)

            metadata[key] = {
                "shape": list(tensor.shape),
                "dtype": "float32",
                "offset": bin_file.tell() - np_arr.nbytes
            }

    # 写入元数据文件
    with open(f"{output_prefix}_metadata.json", "w") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print(f"✅ 权重已导出为二进制文件: {output_prefix}.bin")
    print(f"✅ 元数据已保存为: {output_prefix}_metadata.json")


save_weights_to_bin(model)