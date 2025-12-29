import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings


warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


train_file = "d:\\HuaweiMoveData\\Users\\86189\\Desktop\\Uni-LAP-main_desktop\\SCM\\datasets\\cail\\train_filtered_cail.pkl"
test_file = "d:\\HuaweiMoveData\\Users\\86189\\Desktop\\Uni-LAP-main_desktop\\SCM\\datasets\\cail\\test_filtered_cail.pkl"


output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)


def load_pkl_file(file_path):
    print(f"正在加载文件: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"加载成功! 数据类型: {type(data).__name__}")
        return data
    except Exception as e:
        print(f"加载失败: {e}")
        return None


def analyze_data_structure(data, data_name):
    print(f"\n分析{data_name}数据结构:")
    
    if isinstance(data, dict):
        print(f"数据是字典，包含 {len(data)} 个键:")
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                size = len(value) if isinstance(value, list) else value.shape
                print(f"  - {key}: {type(value).__name__}，大小: {size}")
            else:
                print(f"  - {key}: {type(value).__name__}")
        return list(data.keys())
    elif isinstance(data, list):
        print(f"数据是列表，包含 {len(data)} 个元素")
        if data:
            print(f"第一个元素类型: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                print(f"第一个元素的键: {list(data[0].keys())}")
                return list(data[0].keys())
    return []

# 可视化标签分布
def plot_label_distribution(data, label_key, title, filename):
    if isinstance(data, dict):
        if label_key in data:
            labels = data[label_key]
        else:
            print(f"标签键 '{label_key}' 不存在于字典中")
            return
    elif isinstance(data, list):
        labels = [item[label_key] for item in data if label_key in item]
    else:
        print(f"不支持的数据类型: {type(data).__name__}")
        return
    
    # 统计每个标签出现的次数
    if isinstance(labels[0], (list, np.ndarray)):
        # 多标签情况
        all_labels = []
        for label_list in labels:
            all_labels.extend(label_list)
        label_counts = Counter(all_labels)
    else:
        # 单标签情况
        label_counts = Counter(labels)
    
    # 获取前20个最常见的标签
    top_labels = label_counts.most_common(20)
    if not top_labels:
        print(f"没有找到标签数据")
        return
    
    label_names, counts = zip(*top_labels)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(counts), y=list(label_names), palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('数量', fontsize=12)
    plt.ylabel('标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"已保存标签分布可视化: {filename}")

# 可视化文本长度分布
def plot_text_length_distribution(data, text_key, title, filename):
    if isinstance(data, dict):
        if text_key in data:
            texts = data[text_key]
        else:
            print(f"文本键 '{text_key}' 不存在于字典中")
            return
    elif isinstance(data, list):
        texts = [item[text_key] for item in data if text_key in item]
    else:
        print(f"不支持的数据类型: {type(data).__name__}")
        return
    
    # 计算文本长度
    if isinstance(texts[0], str):
        lengths = [len(text) for text in texts]
    elif isinstance(texts[0], (list, np.ndarray)):
        lengths = [len(text) for text in texts]
    else:
        print(f"不支持的文本类型: {type(texts[0]).__name__}")
        return
    
    plt.figure(figsize=(12, 8))
    sns.histplot(lengths, bins=50, kde=True, color='skyblue')
    plt.title(title, fontsize=16)
    plt.xlabel('文本长度', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"已保存文本长度分布可视化: {filename}")

# 可视化样本数量对比
def plot_dataset_size_comparison(train_data, test_data, title, filename):

    if isinstance(train_data, dict):

        train_size = len(next(iter(train_data.values()))) if train_data else 0
    elif isinstance(train_data, list):
        train_size = len(train_data)
    else:
        train_size = 0
    
    if isinstance(test_data, dict):
        test_size = len(next(iter(test_data.values()))) if test_data else 0
    elif isinstance(test_data, list):
        test_size = len(test_data)
    else:
        test_size = 0
    
    # 绘制对比图
    plt.figure(figsize=(8, 6))
    datasets = ['训练集', '测试集']
    sizes = [train_size, test_size]
    colors = ['lightgreen', 'lightcoral']
    
    sns.barplot(x=datasets, y=sizes, palette=colors)
    plt.title(title, fontsize=16)
    plt.ylabel('样本数量', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"已保存数据集大小对比可视化: {filename}")

# 主函数
def main():
    print("="*60)
    print("CAIL数据集可视化工具")
    print("="*60)
    
    # 加载数据
    train_data = load_pkl_file(train_file)
    test_data = load_pkl_file(test_file)
    
    if not train_data and not test_data:
        print("无法加载数据，程序终止")
        return
    

    train_keys = analyze_data_structure(train_data, "训练集")
    test_keys = analyze_data_structure(test_data, "测试集")
    

    all_keys = list(set(train_keys + test_keys))
    print(f"\n数据中所有可能的键: {all_keys}")
    

    plot_dataset_size_comparison(train_data, test_data, "训练集与测试集样本数量对比", "dataset_size_comparison.png")
    

    common_label_keys = ['labels', 'label', 'article', 'articles', 'law', 'laws']
    common_text_keys = ['text', 'fact', 'content', 'facts', 'description']
    

    label_key = None
    for key in common_label_keys:
        if key in all_keys:
            label_key = key
            break
    

    text_key = None
    for key in common_text_keys:
        if key in all_keys:
            text_key = key
            break

    if label_key:
        print(f"\n使用 '{label_key}' 作为标签键")
        if train_data:
            plot_label_distribution(train_data, label_key, "训练集标签分布", "train_label_distribution.png")
        if test_data:
            plot_label_distribution(test_data, label_key, "测试集标签分布", "test_label_distribution.png")
    else:
        print(f"\n未找到合适的标签键，请手动指定")

    if text_key:
        print(f"\n使用 '{text_key}' 作为文本键")
        if train_data:
            plot_text_length_distribution(train_data, text_key, "训练集文本长度分布", "train_text_length.png")
        if test_data:
            plot_text_length_distribution(test_data, text_key, "测试集文本长度分布", "test_text_length.png")
    else:
        print(f"\n未找到合适的文本键，请手动指定")
    
    print("\n" + "="*60)
    print("可视化完成! 图表已保存到 'visualizations' 目录")
    print("="*60)

if __name__ == "__main__":
    main()