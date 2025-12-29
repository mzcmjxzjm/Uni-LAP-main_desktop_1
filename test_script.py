import subprocess
import sys

# 运行原始脚本
script_path = 'd:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\llm\step3_llm_main_random_version_Call_twice_crime-bert.py'
result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

# 打印输出
print("=== 标准输出 (STDOUT) ===")
print(result.stdout)

print("=== 标准错误 (STDERR) ===")
print(result.stderr)

print("=== 返回码 (Return Code) ===")
print(result.returncode)