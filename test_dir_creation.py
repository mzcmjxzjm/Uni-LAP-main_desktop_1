import os
import datetime

# 测试目录创建
test_dir = os.path.join('d:\\HuaweiMoveData\\Users\\86189\\Desktop\\Uni-LAP-main_desktop\\SCM\\results\\cail\\legal-bert', f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
os.makedirs(test_dir)
print(f'Created directory: {test_dir}')
os.rmdir(test_dir)
print('Directory removed successfully')
print('Directory creation test passed!')