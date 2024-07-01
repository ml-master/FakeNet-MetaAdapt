
import json
from pathlib import Path

# 指定JSON文件路径
input_file_path = Path('D:\workspace\ML\MetaAdapt\data\GossipCop_Styled_Based\gossipcop_v3-1_style_based_fake.json')  # 替换为你的JSON文件路径
output_dir = Path('D:\workspace\ML\MetaAdapt\data\GossipCop_Styled_Based')  # 输出目录

# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

# 读取原始JSON文件
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 准备两个新文件的字典
real_data = {}
fake_data = {}

# 遍历原始数据，根据'label'字段分类
for key, value in data.items():
    if value.get('label') == 'legitime':
        real_data[key] = value
    elif value.get('label') == 'fake':
        fake_data[key] = value

# 写入文件的函数
def write_data_to_json(data_dict, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)

# 写入real.json
real_output_path = output_dir / 'real.json'
write_data_to_json(real_data, real_output_path)
print(f"Real data written to: {real_output_path}")

# 写入fake.json
fake_output_path = output_dir / 'fake.json'
write_data_to_json(fake_data, fake_output_path)
print(f"Fake data written to: {fake_output_path}")