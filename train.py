from config import *
from data_parallel import BalancedDataParallel
from utils import *
from model import make_model
from data_loader import *
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR #学习率调度器，允许用户通过自定义的lambda函数动态调整优化器的学习率
from torch import nn
import torch

# 自定义调整策略
def lr_lambda_fn(step, wramup):
    lr = 0
    if step <= wramup:
        lr = step / wramup * 10
    else:
        lr = wramup / step * 10
    return max(lr, 0.1)

# 每一轮单独处理
def run_epoch(loader, model, loss_fn, optimizer=None):
    # 初始化loss值，和batch数量总数
    total_batchs = 0.
    total_loss = 0.
    # 加载数据并开始训练
    for src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text in loader:

        src_x = src_x.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        tgt_x = tgt_x.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        tgt_y = tgt_y.to(DEVICE)

        output = model(src_x, src_mask, tgt_x, tgt_mask)

        # 交叉熵损失，要求目标值是一维的
        # output.shape[-1] 表示保留 output 张量最后一个维度的大小不变，其他维度的大小全部展平
        loss = loss_fn(output.reshape(-1, output.shape[-1]), tgt_y.reshape(-1))# reshape(-1) 用于将多维张量展平为一维张量

        # 累积batch数量和loss值
        total_batchs += 1
        total_loss += loss.item()

        # 如果有优化器，则表示需要反向传播
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 返回epoch的平均loss值
    return total_loss / total_batchs

# 定义评估函数
def evaluate(loader, model, max_len=50): #限制最大长度50
    model.eval()
    tgt_sent = [] #目标句子
    prob_sent = []#预测句子

    for src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text in loader:
        src_x = src_x.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        batch_prob_text = batch_greedy_decode(model, src_x, src_mask, max_len)

        src_x = src_x.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        tgt_sent += tgt_text 
        prob_sent += batch_prob_text

    print(prob_sent)
    print(tgt_sent)
        
    # 注意参考句子是多组
    return bleu_score(prob_sent, [tgt_sent])



if __name__ == '__main__':
    en_id2vocab, _  = get_vocab('en')
    zh_id2vocab, _ = get_vocab('zh')

    SRC_VOCAB_SIZE = len(en_id2vocab) # 参数动态获取 词表大小
    TGT_VOCAB_SIZE = len(zh_id2vocab)

    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_HEAD, D_FF, N, DROPOUT)
    model = model.to(DEVICE)
    # 多GPU训练
    if MULTI_GPU:
        # model = nn.DataParallel(model)
        model = BalancedDataParallel(BATCH_SIZE_GPU0, model, dim=0)

    # 查看模型结构和参数量
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}") # 44135955参数量

    # 损失函数和优化器
    loss_fn = CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)
    optimizer = Adam(model.parameters(), lr=LR)

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda_fn(step, EPOCH/4))

    best_bleu = 0

    for e in range(EPOCH):
        # 训练流程
        model.train()
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer)
        lr_scheduler.step() #每一轮训练后，调整学习率

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)
        
        # 验证流程
        model.eval()
        dev_loss = run_epoch(dev_loader, model, loss_fn, None)
        dev_bleu = evaluate(dev_loader, model, MAX_LEN)

        print('>> epoch:', e, 'train_loss:', round(train_loss, 6), 'dev_loss:', round(dev_loss, 6), 'dev_bleu:', dev_bleu)
        # print(f"Epoch {e+1}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
            
        # 调用
        print_memory()
        print('--' * 10)

        if dev_bleu > best_bleu:

            model_mod = model.module if MULTI_GPU else model
            torch.save(model_mod.state_dict(), MODEL_DIR+'/best_model.pth')
            best_bleu = dev_bleu








