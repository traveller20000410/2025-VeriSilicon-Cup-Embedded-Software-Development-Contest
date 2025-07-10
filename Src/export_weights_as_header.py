import torch
import numpy as np
from model import CNN

# 加载模型
device = torch.device("cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("model/voice_detector_trained_968_967.pth", map_location=device))
# model.load_state_dict(torch.load("model/vad_cnn_mel_trained.pth", map_location=device))
model.eval()

# 导出函数：将张量转换为 C 数组字符串，每 15 个数值换行
def tensor_to_c_array(name, tensor):
    flat = tensor.detach().numpy().flatten()
    lines = []
    for i in range(0, len(flat), 15):
        line = ", ".join([f"{v:.6f}f" for v in flat[i:i + 15]])
        lines.append(line)
    values = ",\n".join(lines)
    return f"const float {name}[] = {{\n{values}\n}};\n"

# 打开头文件进行写入
with open("weights.h", "w") as f:
    f.write("// Auto-generated weights from PyTorch model\n\n")
    # 导出所有参数，包括 running_mean 和 running_var
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        c_name = key.replace('.', '_')  # 替换掉 . 避免 C 标识符错误
        f.write(tensor_to_c_array(c_name, value))

print("✅ 权重已成功导出到 weights.h，包含 running_mean 和 running_var")