
import json
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torch.optim as optim
import datetime
import math
from typing import List
from transformers import BertConfig, BertModel

import argparse
import copy
import datetime
import os
import pickle
import sys
import time
import random
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from utils.optim import ScheduledOptim
from transformers import AutoTokenizer, BertConfig

import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score
import matplotlib.pyplot as plt


class Attention(nn.Module):
    """注意力机制模块"""

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, attention_mask=None):
        """
        Args:
            rnn_out: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        Returns:
            attended_out: [batch_size, hidden_dim]
        """
        # 计算注意力权重
        attention_weights = self.attention(rnn_out).squeeze(-1)  # [batch_size, seq_len]


        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float("-inf")
            )

        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]

        # 加权求和
        attended_out = torch.bmm(attention_weights.unsqueeze(1), rnn_out).squeeze(
            1
        )  # [batch_size, hidden_dim]

        return attended_out


class LawModel(nn.Module):
    def __init__(self, config):
        super(LawModel, self).__init__()
        self.config = config


        self.embedding = nn.Embedding(config.vocab_size, config.word_emb_dim)


        if config.pretrain_word_embedding is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(config.pretrain_word_embedding)
            )

            if config.HP_freeze_word_emb:
                self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout(config.HP_dropout)


        self.rnn = nn.GRU(
            input_size=config.word_emb_dim,
            hidden_size=config.hidden_size,  
            num_layers=config.num_layers, 
            batch_first=True, 
            bidirectional=config.bidirectional,  
            dropout=config.HP_dropout if config.num_layers > 1 else 0,  
        )

        # 注意力机制
        rnn_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.attention = Attention(rnn_output_size)


        self.fc = nn.Sequential(
            nn.Linear(rnn_output_size, config.mlp_size),
            nn.BatchNorm1d(config.mlp_size),
            nn.ReLU(),
            nn.Dropout(config.HP_dropout),
            nn.Linear(config.mlp_size, config.mlp_size // 2),
            nn.BatchNorm1d(config.mlp_size // 2),
            nn.ReLU(),
            nn.Dropout(config.HP_dropout),
            nn.Linear(config.mlp_size // 2, config.mlp_size // 4),
            nn.BatchNorm1d(config.mlp_size // 4),
            nn.ReLU(),
            nn.Dropout(config.HP_dropout),
            nn.Linear(config.mlp_size // 4, config.law_label_size),
        )

        # 损失函数
        self.law_loss = torch.nn.BCEWithLogitsLoss()  
        self.topk = config.topk  

    def calculate_complexity_features(self, input_facts, attention_mask_list):

        batch_size = input_facts.size(0)
        k_values = []

        for i in range(batch_size):

            if attention_mask_list is not None:
                text_length = attention_mask_list[i].sum().item()
            else:
                text_length = (input_facts[i] != 0).sum().item()

            if text_length < 200:  
                k = 2
            elif text_length < 400: 
                k = 3
            else:  
                k = 4

            k_values.append(k)

        return torch.tensor(k_values, device=input_facts.device, dtype=torch.long)

    def calculate_topk_loss(self, law_probs, law_labels, k_values=None):
        """
        计算Top-K损失，支持动态K值
        Args:
            law_probs: [batch_size, law_label_size]
            law_labels: [batch_size, law_label_size]
            k_values: [batch_size] 每个样本的K值，如果为None则使用默认topk
        """
        batch_size = law_probs.size(0)
        losses = []

        for i in range(batch_size):
            k = k_values[i].item() if k_values is not None else self.topk
            k = min(k, law_probs.size(1))  # 确保k不超过标签数


            topk_probs, topk_indices = torch.topk(
                law_probs[i : i + 1], k, dim=1
            )  # [1, k]


            topk_mask = torch.zeros_like(
                law_probs[i : i + 1], device=law_probs.device
            )  # [1, law_label_size]
            topk_mask.scatter_(1, topk_indices, 1)


            non_topk_probs = law_probs[i : i + 1] * (1 - topk_mask)


            forgotten_mask = (law_labels[i : i + 1] * non_topk_probs).sum(dim=1)  # [1]

            losses.append(forgotten_mask)

        # 计算平均损失
        loss = torch.cat(losses).mean()
        return loss

    def classifier_layer(self, doc_out, law_labels):

        law_logits = self.fc(doc_out) 
        law_loss = self.law_loss(law_logits, law_labels.float())  
        law_probs = torch.sigmoid(law_logits)  


        return law_probs, law_loss

    def forward(self, input_facts, type_ids_list, attention_mask_list, law_labels):

        batch_size = input_facts.size(0)


        input_facts_emb = self.embedding(
            input_facts
        )  


        input_facts_emb = self.embedding_dropout(input_facts_emb)


        rnn_out, _ = self.rnn(
            input_facts_emb
        ) 


        doc_out = self.attention(
            rnn_out, attention_mask_list
        )  

        law_probs, law_loss = self.classifier_layer(
            doc_out, law_labels
        )  
        k_values = self.calculate_complexity_features(input_facts, attention_mask_list)
        topk_loss = self.calculate_topk_loss(law_probs, law_labels, k_values)

        law_preds = (law_probs > 0.3).int()  

        return law_loss, topk_loss, law_preds

    def predict(self, input_facts, type_ids_list, attention_mask_list, law_labels):

        batch_size = input_facts.size(0)

 
        input_facts_emb = self.embedding(
            input_facts
        ) 


        input_facts_emb = self.embedding_dropout(input_facts_emb)

 
        rnn_out, _ = self.rnn(
            input_facts_emb
        ) 


        doc_out = self.attention(
            rnn_out, attention_mask_list
        ) 
        # 分类器
        law_probs, law_loss = self.classifier_layer(
            doc_out, law_labels
        )  

        return law_probs


os.chdir("/home/zjm/zjm_imortant/Uni-LAP-main_desktop_change/SCM")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_rand(SEED_NUM):
    torch.random.manual_seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)
    torch.cuda.manual_seed_all(SEED_NUM)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    def __init__(self):
        self.topk = 3 
        self.hidden_size = 512 
        self.num_layers = 3 
        self.bidirectional = True  

        self.vocab_size = 21128
        self.MAX_SENTENCE_LENGTH = 500
        self.word_emb_dim = 200
        self.pretrain_word_embedding = None
        self.word2id_dict = None
        self.id2word_dict = None
        self.bert_path = None

        self.law_label_size = 74  # TODO law_label_size
        self.law_relation_threshold = 0.3

        self.sent_len = 100
        self.doc_len = 15
        #  hyperparameters
        self.HP_iteration = 1  
        self.HP_batch_size = 32
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5 
        self.HP_gpu = False
        self.HP_lr = 0.001 
        self.HP_lr_decay = 0.1  
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.bert_hidden_size = 768
        self.HP_freeze_word_emb = True

        # L2正则化权重衰减
        self.weight_decay = 1e-5

        # optimizer
        self.use_adam = True
        self.use_adamw = False  # 新增AdamW优化器选项
        self.use_bert = False
        self.use_sgd = False
        self.use_adadelta = False
        self.use_warmup_adam = False
        self.mode = "train"

        self.save_model_dir = ""
        self.save_dset_dir = ""

        self.confused_matrix = None
        self.mlp_size = 1024  # MLP层大小（从512增加到1024）

        self.seed = 10

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Bert Path:           %s" % (self.bert_path))
        print("     Law label     size:  %s" % (self.law_label_size))

        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))

        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Seed                :  %s" % (self.seed))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def load_word_pretrain_emb(self, emb_path):
        self.pretrain_word_embedding = np.cast[np.float32](np.load(emb_path))
        self.word_emb_dim = self.pretrain_word_embedding.shape[1]
        # 根据预训练词向量的实际大小调整vocab_size
        self.vocab_size = self.pretrain_word_embedding.shape[0]
        print("word embedding size:", self.pretrain_word_embedding.shape)
        print(f"更新后的vocab_size: {self.vocab_size}")


class BERTDataset(Dataset):
    def __init__(
        self, data, tokenizer, max_len, id2word_dict, fact_type, law_label_size=74
    ):  # TODO law_label_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.id2word_dict = id2word_dict
        self.word2id_dict = {v: k for k, v in id2word_dict.items()}  # 添加word2id_dict
        self.fact_type = fact_type
        self.law_label_size = law_label_size  # 法条总数

    def __len__(self):
        if self.fact_type == "fact":
            return len(self.data["raw_facts_list"])

    def _convert_ids_to_sent(self, fact):

        mask = np.array(fact) == 164672
        mask = ~mask
        seq_len = mask.sum(1)  # [max_sent_num]
        sent_num_mask = seq_len == 0
        sent_num_mask = ~sent_num_mask
        sent_num = sent_num_mask.sum(0)
        raw_text = []
        for s_idx in range(sent_num):
            cur_seq_len = seq_len[s_idx]
            raw_text.extend(fact[s_idx][:cur_seq_len])

        return [self.id2word_dict[ids] for ids in raw_text]

    def __getitem__(self, index):
        raw_fact_list = self.data["raw_facts_list"][index]
        law_label_lists = self.data["law_label_lists"][index]


        fact_indices = []
        for char in raw_fact_list:
            if char in self.word2id_dict:
                fact_indices.append(self.word2id_dict[char])
            else:
                fact_indices.append(1) 


        law_label_vector = torch.zeros(self.law_label_size, dtype=torch.float)
        for law_id in law_label_lists:
            law_label_vector[law_id] = 1.0 

        return fact_indices, law_label_vector

    def collate_bert_fn(self, batch):
        batch_raw_fact_list, batch_law_label_lists = [], []

        for item in batch:
            batch_raw_fact_list.append(item[0])
            batch_law_label_lists.append(item[1])


        max_len = self.max_len
        padded_input_ids = []
        padded_attention_mask = []


        vocab_size = 21128

        for fact in batch_raw_fact_list:

            fact = [
                int(idx) if int(idx) < vocab_size else vocab_size - 1 for idx in fact
            ]


            fact_tensor = torch.tensor(fact, dtype=torch.long)

            if len(fact_tensor) > max_len:
                padded_fact = fact_tensor[:max_len]
                attention_mask = torch.ones(max_len, dtype=torch.long)
            else:
                padding = torch.zeros(max_len - len(fact_tensor), dtype=torch.long)
                padded_fact = torch.cat([fact_tensor, padding])
                attention_mask = torch.cat(
                    [
                        torch.ones(len(fact_tensor), dtype=torch.long),
                        torch.zeros(len(padding), dtype=torch.long),
                    ]
                )
            padded_input_ids.append(padded_fact)
            padded_attention_mask.append(attention_mask)

        padded_input_ids = torch.stack(padded_input_ids)
        padded_attention_mask = torch.stack(padded_attention_mask)
        # 对于RNN模型，token_type_ids可以设为全0
        padded_token_type_ids = torch.zeros_like(padded_input_ids)
        padded_law_label_lists = torch.stack(batch_law_label_lists)  # 将标签堆叠为张量

        return (
            padded_input_ids,
            padded_token_type_ids,
            padded_attention_mask,
            batch_raw_fact_list,
            padded_law_label_lists,
        )


def load_dataset(path):
    train_path = os.path.join(path, "train_filtered_cail.pkl")
    valid_path = os.path.join(path, "valid_filtered_cail.pkl")
    test_path = os.path.join(path, "test_filtered_cail.pkl")

    train_dataset = pickle.load(open(train_path, mode="rb"))
    valid_dataset = pickle.load(open(valid_path, mode="rb"))
    test_dataset = pickle.load(open(test_path, mode="rb"))

    print("train dataset sample:", train_dataset["raw_facts_list"][0])
    print("train dataset sample len:", len(train_dataset["law_label_lists"]))
    return train_dataset, valid_dataset, test_dataset


def str2bool(params):
    return True if params.lower() == "true" else False


def save_data_setting(data: Config, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    ## save data settings
    with open(save_file, "wb") as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def load_data_setting(save_file):
    with open(save_file, "rb") as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def get_result(law_target, law_preds, mode):
    """
    计算多标签分类任务的评估指标。
    :param law_target: 真实标签，二维列表或数组（样本数 × 法条数）
    :param law_preds: 预测标签，二维列表或数组（样本数 × 法条数）
    :param mode: 评估模式（用于打印日志）
    :return: 宏平均 F1 分数
    """
    # 计算宏平均 F1、Precision 和 Recall
    law_macro_f1 = f1_score(law_target, law_preds, average="macro", zero_division=0)
    law_macro_precision = precision_score(
        law_target, law_preds, average="macro", zero_division=0
    )
    law_macro_recall = recall_score(
        law_target, law_preds, average="macro", zero_division=0
    )
    law_accuracy_score = accuracy_score(law_target, law_preds)

    # 计算 Hamming Loss 和 Jaccard Similarity
    hamming = hamming_loss(law_target, law_preds)
    jaccard = jaccard_score(law_target, law_preds, average="samples", zero_division=0)

    print(f"Law task ({mode}):")
    print(
        f"@ACC: {law_accuracy_score:.4f} Macro F1: {law_macro_f1:.4f}, Macro Precision: {law_macro_precision:.4f}, Macro Recall: {law_macro_recall:.4f}"
    )
    print(f"@Hamming Loss: {hamming:.4f}, Jaccard Similarity: {jaccard:.4f}")

    return law_macro_f1


def calculate_topk_accuracy(true_labels, pred_probs, k):
    """
    Args:
        true_labels (list): 真实标签，形状为 [batch_size, num_classes]。
        pred_probs (list): 预测概率，形状为 [batch_size, num_classes]。
        k (int): Top-K 的 K 值。

    Returns:
        float:
    """
    true_labels = torch.tensor(true_labels)  # 转换为张量
    pred_probs = torch.tensor(pred_probs)  # 转换为张量

    _, topk_preds = torch.topk(pred_probs, k, dim=1)  # 形状为 [batch_size, k]

    correct = 0  # 统计所有样本中真实标签在 Top-K 预测中的累计数量
    total_labels = 0  # 统计所有样本的真实标签总数

    for i in range(true_labels.size(0)):
        true_label = torch.nonzero(true_labels[i]).squeeze()  # 真实标签的索引
        if true_label.dim() == 0:  # 如果只有一个真实标签
            true_label = true_label.unsqueeze(0)

        total_labels += true_label.size(0)

        for label in true_label:
            if label in topk_preds[i]:
                correct += 1

    accuracy = correct / total_labels

    return accuracy


def calculate_topk_accuracy_accurate(true_labels, pred_probs, k):
    """
    Args:
        true_labels (list): 真实标签，形状为 [batch_size, num_classes]。
        pred_probs (list): 预测概率，形状为 [batch_size, num_classes]。
        k (int): Top-K 的 K 值。

    Returns:
        float: 所有样本的 Top-K 准确率均值。
    """
    true_labels = torch.tensor(true_labels)  # 转换为张量
    pred_probs = torch.tensor(pred_probs)  # 转换为张量

    _, topk_preds = torch.topk(pred_probs, k, dim=1)  # 形状为 [batch_size, k]

    sample_accuracies = []  # 存储每个样本的 Top-K 准确率

    for i in range(true_labels.size(0)):
        true_label = torch.nonzero(true_labels[i]).squeeze()  # 真实标签的索引
        if true_label.dim() == 0:  # 如果只有一个真实标签
            true_label = true_label.unsqueeze(0)

        # 计算当前样本的 Top-K 准确率
        correct = 0
        for label in true_label:
            if label in topk_preds[i]:
                correct += 1

        # 当前样本的准确率 = 正确预测的标签数 / 真实标签数
        sample_accuracy = correct / true_label.size(0)
        sample_accuracies.append(sample_accuracy)

    # 计算所有样本的 Top-K 准确率均值
    mean_accuracy = torch.tensor(sample_accuracies).mean().item()

    return mean_accuracy


def evaluate(model, valid_dataloader, name, epoch_idx, k=3):
    """
    评估模型在多标签分类任务上的性能，并计算 Top-K 准确率。

    Args:
        model: 模型
        valid_dataloader: 验证集 DataLoader
        name: 评估名称（用于打印日志）
        epoch_idx: 当前 epoch 索引
        k: Top-K 的 K 值，默认为 5

    Returns:
        macro_f1: 宏平均 F1 分数
        topk_accuracy: Top-K 准确率
    """
    model.eval()
    ground_law_y = []  # 真实标签
    predicts_law_y = []  # 预测标签
    all_law_probs = []  # 存储所有预测概率

    print_num = 0

    for batch_idx, datapoint in enumerate(valid_dataloader):
        fact_list, type_ids_list, attention_mask_list, _, law_label_lists = datapoint

        # 获取模型设备
        device = next(model.parameters()).device

        # 将数据移动到模型所在设备
        fact_list = fact_list.to(device)
        type_ids_list = type_ids_list.to(device)
        attention_mask_list = attention_mask_list.to(device)
        law_label_lists = law_label_lists.to(device)

        # 获取模型预测结果
        law_probs = model.predict(
            fact_list, type_ids_list, attention_mask_list, law_label_lists
        )
        all_law_probs.extend(law_probs.cpu().tolist())  # 存储预测概率

        law_preds = (law_probs > 0.3).int()  # 阈值进行预测

        # 收集真实标签和预测标签
        ground_law_y.extend(law_label_lists.cpu().tolist())
        predicts_law_y.extend(law_preds.cpu().tolist())
        print_num += 1

    macro_f1 = get_result(ground_law_y, predicts_law_y, name)

    topk_accuracy = calculate_topk_accuracy(ground_law_y, all_law_probs, k)
    print(f"{name} Top-{k} Accuracy: {topk_accuracy:.4f}")
    topk_accuracy_accu = calculate_topk_accuracy_accurate(
        ground_law_y, all_law_probs, k
    )
    print(f"{name} Top-{k} Accuracy_accurate: {topk_accuracy_accu:.4f}")

    if name == "Test":
        # 将所有的 law_probs 存储到 JSON 文件中
        with open(f"{config.save_model_dir}/law_probs.json", "w") as f:
            json.dump(all_law_probs, f)
        print("已将 law_probs 存储到 JSON 文件中, 位置在：", config.save_model_dir)

    return topk_accuracy


def train(model, dataset, config: Config):
    train_dataloader = dataset["train_data_set"]
    valid_dataloader = dataset["valid_data_set"]
    test_dataloader = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
    print("Training model...")
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if config.use_warmup_adam:
        optimizer = ScheduledOptim(
            optim.Adam(
                parameters,
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=config.weight_decay,
            ),
            d_model=256,
            n_warmup_steps=2000,
        )
    elif config.use_adamw:
        # 使用AdamW优化器，提供更好的权重衰减实现
        optimizer = optim.AdamW(
            parameters, lr=config.HP_lr, weight_decay=config.weight_decay
        )
    elif config.use_sgd:
        optimizer = optim.SGD(
            parameters,
            lr=config.HP_lr,
            momentum=config.HP_momentum,
            weight_decay=config.weight_decay,
        )
    elif config.use_adam:
        # 在Adam中添加L2正则化
        optimizer = optim.Adam(
            parameters, lr=config.HP_lr, weight_decay=config.weight_decay
        )
    elif config.use_bert:
        print("optimizer use_bert!")
        optimizer = optim.Adam(
            parameters, lr=5e-6, weight_decay=config.weight_decay
        )  # fine tuning
    else:
        raise ValueError("Unknown optimizer")
    print("optimizer: ", optimizer)

    # 初始化最佳性能指标
    best_score = -float("inf")  # 假设性能指标是越大越好（例如 F1 分数）
    best_epoch = -1

    # 添加调试信息
    print("准备开始训练循环...")
    print(f"训练数据加载器长度: {len(train_dataloader)}")
    print(f"每个批次大小: {config.HP_batch_size}")
    print(f"总迭代次数: {config.HP_iteration}")

    for idx in range(config.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.HP_iteration))
        sample_loss = 0
        sample_law_loss = 0
        sample_topk_loss = 0

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size

        ground_law_y, predicts_law_y = [], []

        # 为每个epoch初始化指标记录列表
        epoch_iteration_list = []
        epoch_total_loss_list = []
        epoch_law_loss_list = []
        epoch_topk_loss_list = []
        epoch_hamming_loss_list = []
        epoch_jaccard_score_list = []

        print(f"开始处理 Epoch {idx} 的数据批次...")
        for batch_idx, datapoint in enumerate(train_dataloader):
            print(f"处理批次: {batch_idx+1}/{len(train_dataloader)}")
            (
                fact_list,
                type_ids_list,
                attention_mask_list,
                _,
                law_label_lists,
            ) = datapoint
            # 根据配置将数据移动到相应设备
            if config.HP_gpu:
                fact_list = fact_list.to(DEVICE)
                type_ids_list = type_ids_list.to(DEVICE)
                attention_mask_list = attention_mask_list.to(DEVICE)
                law_label_lists = law_label_lists.to(DEVICE)

            law_loss, topk_loss, law_preds = model.forward(
                fact_list, type_ids_list, attention_mask_list, law_label_lists
            )

            loss = law_loss + topk_loss * 0.1  # 超参

            sample_loss += loss.data
            sample_law_loss += law_loss.data
            sample_topk_loss += topk_loss.data

            # 将 logits 转换为二进制向量
            ground_law_y.extend(law_label_lists.cpu().tolist())
            predicts_law_y.extend(law_preds.cpu().tolist())

            if (batch_idx + 1) % 100 == 0:
                # 计算 Hamming Loss 和 Jaccard Similarity
                cur_hamming = hamming_loss(ground_law_y, predicts_law_y)
                cur_jaccard = jaccard_score(
                    ground_law_y, predicts_law_y, average="samples", zero_division=0
                )

                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time

                print(
                    "Instance: %s; Time: %.2fs; total loss: %.2f; law loss: %.2f; top-k loss: %.2f; Hamming Loss: %.4f; Jaccard: %.4f"
                    % (
                        (batch_idx + 1),
                        temp_cost,
                        sample_loss,
                        sample_law_loss,
                        sample_topk_loss,
                        cur_hamming,
                        cur_jaccard,
                    )
                )

                # 记录当前指标
                epoch_iteration_list.append(batch_idx + 1)
                epoch_total_loss_list.append(
                    sample_loss.item() / 100.0
                )  # 平均每个batch的loss
                epoch_law_loss_list.append(sample_law_loss.item() / 100.0)
                epoch_topk_loss_list.append(sample_topk_loss.item() / 100.0)
                epoch_hamming_loss_list.append(cur_hamming)
                epoch_jaccard_score_list.append(cur_jaccard)

                if (batch_idx + 1) % 1000 == 0:
                    model.eval()
                    current_score = evaluate(model, valid_dataloader, "Valid", -1)
                    model.train()

                sys.stdout.flush()
                sample_loss = 0
                sample_law_loss = 0
                sample_topk_loss = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.HP_clip)
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # 验证集评估
        model.eval()
        current_score = evaluate(model, valid_dataloader, "Valid", -1)
        model.train()
        print(f"Valid current score  : {current_score}")

        # 保存所有模型参数 防止训练断了
        model_name = os.path.join(config.save_model_dir, f"{idx}.ckpt")
        torch.save(model.state_dict(), model_name)

        # 如果当前模型在验证集上的性能更好，则保存模型
        if current_score > best_score:
            best_score = current_score
            best_epoch = idx
            model_name = os.path.join(config.save_model_dir, f"best_model_epoch.ckpt")
            torch.save(model.state_dict(), model_name)
            print(f"New best model saved at epoch {idx} with score {best_score:.4f}")

        # 测试集评估
        model.eval()
        _ = evaluate(model, test_dataloader, "Test", -1)
        model.train()

        # 在当前epoch结束后绘制指标曲线
        if epoch_iteration_list:
            # 创建一个新的图形
            plt.figure(figsize=(12, 8))

            # 创建第一个Y轴（左侧）用于损失指标
            ax1 = plt.gca()

            # 绘制损失曲线
            ax1.plot(
                epoch_iteration_list, epoch_total_loss_list, "b-", label="Total Loss"
            )
            ax1.plot(epoch_iteration_list, epoch_law_loss_list, "r-", label="Law Loss")
            ax1.plot(
                epoch_iteration_list, epoch_topk_loss_list, "g-", label="Top-k Loss"
            )

            # 设置左侧Y轴标签
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel("Loss", color="k")
            ax1.tick_params("y", colors="k")

            # 创建第二个Y轴（右侧）用于评估指标
            ax2 = ax1.twinx()

            # 绘制评估指标曲线
            ax2.plot(
                epoch_iteration_list,
                epoch_hamming_loss_list,
                "y-",
                label="Hamming Loss",
            )
            ax2.plot(
                epoch_iteration_list,
                epoch_jaccard_score_list,
                "m-",
                label="Jaccard Score",
            )

            # 设置右侧Y轴标签
            ax2.set_ylabel("Metric", color="k")
            ax2.tick_params("y", colors="k")

            # 添加标题和图例
            plt.title(f"Training Metrics for Epoch {idx+1}")

            # 合并图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            # 确保图形布局合理
            plt.tight_layout()

            # 保存图形，文件名包含epoch信息
            plot_path = os.path.join(
                config.save_model_dir, f"training_metrics_epoch_{idx+1}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Training metrics plot for epoch {idx+1} saved to {plot_path}")

    print(
        f"Training complete. Best model saved at epoch {best_epoch} with score {best_score:.4f}"
    )


def Test(model, dataset, config: Config):
    test_dataloader = dataset["test_data_set"]
    print(model)
    _ = evaluate(model, test_dataloader, "Test", -1)


if __name__ == "__main__":
    print(datetime.datetime.now())
    BASE = "/home/zjm/zjm_imortant/Uni-LAP-main_desktop_change/SCM"
    parser = argparse.ArgumentParser(description="Uni-LAP")
    parser.add_argument(
        "--data_path",
        default="/home/zjm/zjm_imortant/Uni-LAP-main_desktop_change/SCM/datasets/cail",
    )
    parser.add_argument("--status", default="train")
    parser.add_argument("--savemodel", default=BASE + "/results/cail/RNN")
    parser.add_argument("--loadmodel", default="")

    parser.add_argument("--embedding_path", default=BASE + "/cail_thulac.npy")
    parser.add_argument("--word2id_dict", default=BASE + "/data/w2id_thulac.pkl")

    parser.add_argument("--word_emb_dim", default=200, type=int)
    parser.add_argument("--MAX_SENTENCE_LENGTH", default=510, type=int)

    parser.add_argument("--HP_iteration", default=1, type=int)
    parser.add_argument("--HP_batch_size", default=32, type=int)
    parser.add_argument("--HP_hidden_dim", default=256, type=int)
    parser.add_argument("--HP_dropout", default=0.5, type=float)

    parser.add_argument("--HP_lr", default=1e-3, type=float)
    parser.add_argument("--HP_lr_decay", default=0.1, type=float)
    parser.add_argument("--HP_freeze_word_emb", action="store_true")

    # 优化器选项
    parser.add_argument("--use_adamw", action="store_true", help="使用AdamW优化器")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="L2正则化权重衰减")

    parser.add_argument("--seed", default=2022, type=int)

    # crime-bert xs刑事
    # parser.add_argument('--bert_path', default='/home/u22451152/Uni-LAP/SCM/xs', type=str)
    # 经典bert
    parser.add_argument(
        "--bert_path",
        default="/home/zjm/zjm_imortant/Uni-LAP-main_desktop_change/bert-base-chinese",
        type=str,
    )
    # legal-bert
    # parser.add_argument('--bert_path', default='/home/u22451152/nlpaueb/legal-bert-base-uncased', type=str)

    parser.add_argument("--sample_size", default="all", type=str)

    parser.add_argument("--mlp_size", default=1024, type=int)
    parser.add_argument("--law_relation_threshold", default=0.3, type=float)
    parser.add_argument(
        "--model_path",
        default="/home/zjm/zjm_imortant/Uni-LAP-main_desktop_change/SCM/results/cail/RNN_baseline_0117/zjm1/best_model_epoch.ckpt",
        type=str,
    )
    args = parser.parse_args()
    print(args)

    seed_rand(args.seed)

    status = args.status

    print("New config....")
    config = Config()
    config.HP_iteration = args.HP_iteration
    config.HP_batch_size = args.HP_batch_size
    config.HP_hidden_dim = args.HP_hidden_dim
    config.HP_dropout = args.HP_dropout
    config.HP_lr = args.HP_lr
    config.MAX_SENTENCE_LENGTH = args.MAX_SENTENCE_LENGTH
    config.HP_lr_decay = args.HP_lr_decay
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.save_model_dir = os.path.join(args.savemodel, timestamp_str)
    config.HP_freeze_word_emb = args.HP_freeze_word_emb
    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)

    config.mlp_size = args.mlp_size
    config.use_adamw = args.use_adamw
    config.weight_decay = args.weight_decay
    config.word2id_dict = pickle.load(open(args.word2id_dict, "rb"))
    config.id2word_dict = {item[1]: item[0] for item in config.word2id_dict.items()}
    config.bert_path = args.bert_path
    config.seed = args.seed

    config.load_word_pretrain_emb(args.embedding_path)
    save_data_setting(config, os.path.join(config.save_model_dir, "data.dset"))
    config.show_data_summary()

    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)  # 导入分词器
    train_data, valid_data, test_data = load_dataset(args.data_path)
    if args.sample_size != "all":
        sample_size = int(args.sample_size)
        sampled_train_data = {}
        start = random.randint(0, len(train_data["raw_facts_list"]) - sample_size)
        print("start:", start)
        for k, v in train_data.items():
            sampled_train_data[k] = train_data[k][start : start + sample_size]

        train_data = sampled_train_data

    if status == "train":
        train_dataset = BERTDataset(
            train_data,
            tokenizer,
            config.MAX_SENTENCE_LENGTH,
            config.id2word_dict,
            "fact",
        )
        valid_dataset = BERTDataset(
            valid_data,
            tokenizer,
            config.MAX_SENTENCE_LENGTH,
            config.id2word_dict,
            "fact",
        )
        test_dataset = BERTDataset(
            test_data,
            tokenizer,
            config.MAX_SENTENCE_LENGTH,
            config.id2word_dict,
            "fact",
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.HP_batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_bert_fn,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.HP_batch_size,
            shuffle=False,
            collate_fn=valid_dataset.collate_bert_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.HP_batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_bert_fn,
        )

        print(
            "train_data %d, valid_data %d, test_data %d."
            % (len(train_dataset), len(valid_dataset), len(test_dataset))
        )

        data_dict = {
            "train_data_set": train_dataloader,
            "test_data_set": test_dataloader,
            "valid_data_set": valid_dataloader,
        }

        seed_rand(args.seed)
        model = LawModel(config)
        # model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        # 训练阶段
        print("\nTraining...")
        if config.HP_gpu:
            model.cuda()
        train(model, data_dict, config)

    elif status == "test":
        test_dataset = BERTDataset(
            test_data,
            tokenizer,
            config.MAX_SENTENCE_LENGTH,
            config.id2word_dict,
            "fact",
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.HP_batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_bert_fn,
        )

        print("test_data %d." % (len(test_dataset)))

        data_dict = {"test_data_set": test_dataloader}

        seed_rand(args.seed)
        model = LawModel(config)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        if config.HP_gpu:
            model.cuda()

        print("\nTesting...")
        Test(model, data_dict, config)
