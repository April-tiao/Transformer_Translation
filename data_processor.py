from config import *
import json
from collections import Counter
from utils import *
def generate_vocab():
    # 预设特殊标记词
    en_vocab = ['<pad>', '<unk>', '<sos>', '<eos>']
    zh_vocab = ['<pad>', '<unk>', '<sos>', '<eos>']

    # 加载json文件，并记录分词结果
    zh_vocab_list = []
    en_vocab_list = []
    # 解析json文件
    with open(TRAIN_SAMPLE_PATH, encoding='utf-8') as file:
        lines = json.loads(file.read())
        for en_sent, zh_sent in lines:
            en_vocab_list += divided_en(en_sent)
            zh_vocab_list += divided_zh(zh_sent)
    # print(len(en_vocab_list))
    # print(len(zh_vocab_list))   
    print('train_sample count:', len(lines))

    # 按出现次数，去重生成词表
    # 如果语料库够大 ，可以按最小次数，过滤掉生僻词。
    # 按次数生成词表，此处可以去掉生僻词
    zh_vocab_kv = Counter(zh_vocab_list).most_common()
    zh_vocab += [k.lower() for k,v in zh_vocab_kv]

    en_vocab_kv = Counter(en_vocab_list).most_common()
    en_vocab += [k.lower() for k,v in en_vocab_kv]

    print('en_vocab count:', len(en_vocab))
    print('zh_vocab count:', len(zh_vocab))

    # 生成词表文件
    with open(ZH_VOCAB_PATH, 'w', encoding='utf-8') as file:
        file.write('\n'.join(zh_vocab))
        
    with open(EN_VOCAB_PATH, 'w', encoding='utf-8') as file:
        file.write('\n'.join(en_vocab))



if __name__ == '__main__':
    generate_vocab()
