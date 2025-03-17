from config import *
from utils import *
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence # 用于padding batch数据
import json
import torch

# 加载数据集
class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            file_path = TRAIN_SAMPLE_PATH
        elif type == 'dev':
            file_path = DEV_SAMPLE_PATH
        # 读取文件
        with open(file_path, encoding='utf-8') as file:
            self.lines = json.loads(file.read())
        # 词表引入,拿到索引值
        _, self.en_vocab2id = get_vocab('en')
        _, self.zh_vocab2id = get_vocab('zh')
    def __len__(self):
        return len(self.lines) #返回样本数量
    
    # 单条数据 tokenizer
    # 英文翻译成中文，所以 en_text 是 source，zh_text 是 target，做中文翻译成英文，反过来即可。返回 zh_text 是用于后续做模型评估。
    def __getitem__(self, index):
        en_text, zh_text = self.lines[index]
        source = [self.en_vocab2id.get(v.lower(), UNK_ID) for v in divided_en(en_text)]
        target = [self.zh_vocab2id.get(v.lower(), UNK_ID) for v in divided_zh(zh_text)]
        return source, target, zh_text #zh_text不是tensor结构，所以返回的是list结构
    
    def collate_fn(self, batch):
        """
        自定义collate_fn函数用于数据预处理和格式化。
        
        该函数主要用于处理数据加载阶段的批量数据，包括源序列和目标序列的填充、
        以及相应注意力掩码的生成，以适应模型的训练需求。
        
        参数:
        - batch: 包含一批数据的元组，每个元组包括源序列(batch_src)、目标序列(batch_tgt)和目标文本(tgt_text)。
        
        返回值:
        - src_x: 填充后的源序列张量。
        - src_mask: 源序列的填充掩码。
        - tgt_x: 填充并移位后的目标序列张量（输入到解码器）。
        - tgt_mask: 目标序列的联合掩码，包括填充掩码和后续掩码。
        - tgt_y: 填充后的目标序列张量（解码器的目标输出）。
        - tgt_text: 目标文本列表，用于评估或日志记录。
        """
        # 分离批量数据为源序列、目标序列和目标文本
        batch_src, batch_tgt, tgt_text = zip(*batch)
        
        # source
        # 对源序列进行填充，添加SOS和EOS标记，并转换为LongTensor类型
        src_x = pad_sequence([torch.LongTensor([SOS_ID] + src + [EOS_ID]) for src in batch_src], True, PAD_ID)
        # 生成源序列的填充掩码
        src_mask = get_padding_mask(src_x, PAD_ID) #有padd 的mask为1，没有padd的mask为0
        
        # target
        # 对目标序列进行填充，添加SOS和EOS标记，并转换为LongTensor类型
        tgt_f = pad_sequence([torch.LongTensor([SOS_ID] + tgt + [EOS_ID]) for tgt in batch_tgt], True, PAD_ID)
        # 将目标序列进行移位，作为解码器的输入
        tgt_x = tgt_f[:, :-1] #不要最后一列
        # 生成目标序列的填充掩码
        tgt_pad_mask = get_padding_mask(tgt_x, PAD_ID)
        # 生成目标序列的后续掩码，以防止解码器在训练时看到未来的信息
        tgt_subsqueent_mask = get_subsequent_mask(tgt_x.size(1))
        # 结合填充掩码和后续掩码，生成最终的目标序列掩码
        tgt_mask = tgt_pad_mask | tgt_subsqueent_mask
        # 获取目标序列的预期输出（从tgt_f中剔除SOS标记）
        tgt_y = tgt_f[:, 1:] #不要第一列
        
        # 返回处理后的源序列、目标序列、相应的掩码以及目标文本
        return src_x, src_mask, tgt_x, tgt_mask, tgt_y, tgt_text
    


# 测试
train_dataset = Dataset('train')
# 对训练数据集进行批量处理、打乱顺序和自定义数据整理
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

dev_dataset = Dataset('dev')
# 对验证数据集进行批量处理、不打乱顺序和自定义数据整理
dev_loader = data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dev_dataset.collate_fn)


# 调用测试
# 直接调用会报错，因为 get_item 返回的不是 tensor 结构，而是 list 结构，需要对其进行处理。
if __name__ == '__main__':
    ds = Dataset('dev')
    loader = data.DataLoader(ds, batch_size=2,collate_fn = ds.collate_fn) # batch_size=2 表示每次处理 2 个句子对（源语言句子 + 目标语言句子组成的配对）
    print(next(iter(loader)))
    # 输出结果解释
    #tensor([5,6]),   # 第1个词的ID（两个句子中的第一个词） ‘我’ ‘他’
    # tensor([10,8]),  # 第2个词的ID（两个句子中的第二个词）‘喜欢’ ‘爱’


