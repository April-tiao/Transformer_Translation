import jieba
import re  
from config import *
import sacrebleu


# 中文分词
def divided_zh(sentence):
    return jieba.lcut(sentence)
# 英文分词
def divided_en(sentence):
    # 使用正则表达式匹配单词和标点符号
    pattern = r'\w+|[^\w\s]'
    return re.findall(pattern, sentence)

# 词表解析函数
def get_vocab(lang='en'):
    if lang == 'en':
        file_path = EN_VOCAB_PATH
    elif lang == 'zh':
        file_path = ZH_VOCAB_PATH
    
    with open(file_path, encoding='utf-8') as file:
        lines = file.read()

    id2vocab = lines.split('\n')
    vocab2id = {v:k for k,v in enumerate(id2vocab)} #enumerate(id2vocab) 用于生成词汇表中每个词汇的索引和对应的词汇，从而构建 vocab2id 字典
    return id2vocab, vocab2id


#生成下三角掩码
def get_subsequent_mask(size):
    # 1为batch维度
    mask_shape = (1, size, size)
    # 使用torch.tril函数生成下三角矩阵，然后取反，以得到后续位置为0，当前位置及其前序位置为1的掩码
    # 最后返回这个掩码，用于后续的注意力机制中，以确保模型在预测第i个词时，只能看到第i个词之前的词
    return 1-torch.tril(torch.ones(mask_shape)).byte()

# Attention处理Key_Padding_Mask
def get_padding_mask(x, padding_idx):
    # 扩展Q维度
    return (x == padding_idx).unsqueeze(1).byte()

def bleu_score(hyp, refs):
    bleu = sacrebleu.corpus_bleu(hyp, refs, tokenize='zh')
    return round(bleu.score, 2)

# 逐字生成预测值
def batch_greedy_decode(model, src_x, src_mask, max_len=50):

    model_mod = model.module if MULTI_GPU else model


    zh_id2vocab, _ = get_vocab('zh')
    memory = model_mod.encoder(src_x, src_mask)
    # print(memory.shape) #torch.Size([3, 6, 512]) 3个句子，6个词，512个维度

    # 初始化目标值
    prob_x = torch.tensor([[SOS_ID]] * src_x.size(0)) #.size(0)表示取第0维度的大小：几个句子
    prob_x = prob_x.to(DEVICE)

    for _ in range(max_len): #生成第i个英文词
        prob_mask = get_padding_mask(prob_x, PAD_ID) #逐字生成，不需要上三角掩码
        output = model_mod.decoder(prob_x, prob_mask, memory, src_mask)
        # print(output.shape)#torch.Size([3, 1, 512]) 3个句子，1个词，512个维度

        output = model_mod.generator(output[:, -1, :])#上一层decoder输出，取最后一个词的输出,1个数据

        predict = torch.argmax(output, dim=-1, keepdim=True)
        prob_x = torch.concat([prob_x, predict], dim=-1)

        # 全部预测结束，结束循环
        if torch.all(predict==EOS_ID).item():
            break
    # exit() #循环生产了50个词，这里直接退出

    # tokenizer逆转，生成预测句子
    # 根据预测值id，解析翻译后的句子
    batch_prob_text = []
    for prob in prob_x:
        prob_text = []
        for prob_id in prob:
            if prob_id == SOS_ID:
                continue
            if prob_id == EOS_ID:
                break
            prob_text.append(zh_id2vocab[prob_id])
        batch_prob_text.append(''.join(prob_text)) #文本拼接
    return batch_prob_text

# 观察GPU显存占用情况
def print_memory():
    # 获取当前可用的GPU数量
    num_gpus = torch.cuda.device_count()
    # 遍历每个GPU，输出GPU的占用情况
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_name(i)
        utilization = round(torch.cuda.max_memory_allocated(i) / 1024**3, 2)  # 显存使用量（以GB为单位）
        print(f"GPU {i}: {gpu}, Memory Utilization: {utilization} GB")


# 测试
if __name__ == '__main__':
    # print(divided_zh('我爱北京天安门'))
    # print(divided_en('naver say naver!'))

    # id2vocab, vocab2id = get_vocab('en')
    # print(id2vocab)
    # print(vocab2id)

    target = '我喜欢读书。'
    vocabs = divided_zh(target) # 分词
    # 获取中文词汇表及其对应的ID映射，get_vocab是一个返回词汇表和词汇到ID映射的函数
    _, zh_vocab2id = get_vocab('zh') #只对第二个返回值 vocab2id 感兴趣，并将其赋值给 zh_vocab2id。get_vocab('zh') 函数返回两个值：id2vocab 和 vocab2id
    # 将分词后的词汇转换为对应的ID序列，如果词汇不在词汇表中，则使用未知词汇（UNK）的ID
    tokens = [zh_vocab2id.get(v, UNK_ID) for v in vocabs]
    print(tokens)#转换后的ID序列
