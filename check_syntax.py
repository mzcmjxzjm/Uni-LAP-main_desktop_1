import ast

# 读取原始脚本内容
script_path = 'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\step3_llm_main_random_version_Call_twice_crime-bert.py'

with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查语法
try:
    ast.parse(content)
    print("脚本语法正确！")
    
    # 检查路径格式
    import re
    windows_paths = re.findall(r'"(D:\\[^"]+)"', content)
    if windows_paths:
        print("\n找到的Windows路径：")
        for path in windows_paths:
            print(f"  - {path}")
    
    raw_strings = re.findall(r'r"([^"]+)"', content)
    if raw_strings:
        print("\n找到的原始字符串路径：")
        for path in raw_strings:
            print(f"  - {path}")
            
except SyntaxError as e:
    print(f"语法错误：{e}")
    print(f"行号：{e.lineno}")
    print(f"位置：{e.offset}")
    print(f"错误信息：{e.msg}")
    
    # 打印错误行附近的内容
    lines = content.split('\n')
    start_line = max(0, e.lineno - 3)
    end_line = min(len(lines), e.lineno + 3)
    
    print("\n错误行附近的内容：")
    for i in range(start_line, end_line):
        prefix = ">>> " if i == e.lineno - 1 else "    "
        print(f"{prefix}{i+1}: {lines[i]}")
        if i == e.lineno - 1:
            print(f"    {' ' * (e.offset - 1)}^")
            print(f"    {' ' * (e.offset - 1)}{e.msg}")