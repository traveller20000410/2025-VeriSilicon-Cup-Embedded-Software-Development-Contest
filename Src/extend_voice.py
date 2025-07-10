import os
import shutil
import re

# 设置大目录路径（请替换为你的实际路径）
base_dir = r"C:\Users\22485\Downloads\SHALCAS22A\corpus"  # 例如：r""""
target_dir = os.path.join(base_dir, "extend_set")

# 创建目标文件夹
os.makedirs(target_dir, exist_ok=True)

# 遍历所有子文件夹
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and folder != "extend_set":
        # 检查该文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if "s0" in filename and filename.endswith(".wav"):
                # 提取新文件名所需的信息
                match = re.match(r"(000\d+).*?(s00\d)", filename)
                if match:
                    folder_id = match.group(1)[-3:]  # 取后三位
                    l_code = match.group(2)
                    new_filename = f"{folder_id}_{l_code}.wav"

                    # 复制并改名到目标目录
                    src_path = os.path.join(folder_path, filename)
                    dst_path = os.path.join(target_dir, new_filename)
                    shutil.copyfile(src_path, dst_path)
                    print(f"已复制: {src_path} -> {dst_path}")
