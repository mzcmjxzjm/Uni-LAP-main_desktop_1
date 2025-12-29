import sys
import os

# 将llm目录添加到Python路径
sys.path.insert(0, r'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm')

print("Python路径：")
for path in sys.path:
    print(f"  - {path}")

# 检查导入
try:
    print("\n尝试导入llm.llm_api...")
    from llm.llm_api import APIChat
    print("  ✓ 成功导入llm.llm_api")
except ImportError as e:
    print(f"  ✗ 导入失败：{e}")
    import traceback
    traceback.print_exc()

try:
    print("\n尝试导入prompts_1...")
    import prompts_1
    print("  ✓ 成功导入prompts_1")
except ImportError as e:
    print(f"  ✗ 导入失败：{e}")
    import traceback
    traceback.print_exc()

try:
    print("\n尝试导入utils.utils...")
    from utils.utils import extract_and_parse_json, law_idx2name, get_ecthr_law_name
    print("  ✓ 成功导入utils.utils")
except ImportError as e:
    print(f"  ✗ 导入失败：{e}")
    import traceback
    traceback.print_exc()

# 检查文件是否存在
print("\n检查关键文件是否存在：")

files_to_check = [
    r'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\step3_llm_main_random_version_Call_twice_crime-bert.py',
    r'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\datasets\cail\Probs\Crime-Bert_probs_not_topk\law_name_top3.json',
    r'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\datasets\cail\data\raw_pkl\test_dataset.pkl',
    r'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\datasets\cail\data\law_name_define_0124.txt',
    r'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\datasets\cail\data\law_labels_cail_filtered.txt'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path}")
