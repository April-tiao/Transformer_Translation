from config import *
from utils import *
from model import make_model
from torch.nn.utils.rnn import pad_sequence  # 显式导入 pad_sequence



if __name__ == '__main__':
    
    # 模型实例化，加载模型参数
    en_id2vocab, en_vocab2id  = get_vocab('en')
    zh_id2vocab, zh_vocab2id  = get_vocab('zh')

    SRC_VOCAB_SIZE = len(en_id2vocab)
    TGT_VOCAB_SIZE = len(zh_id2vocab)
    model = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_HEAD, D_FF, N, DROPOUT)
    model = model.to(DEVICE)

    model.load_state_dict(torch.load(MODEL_DIR + '/best_model.pth'), map_location=DEVICE)
    model.eval()
    
    # 构造输入值，并做预测
    texts = [
        "I am a teacher",
        "He likes traveling",
    ]
    batch_src_token = [[en_vocab2id.get(v.lower, UNK_ID) for v in divided_en(text)] for text in texts]
    batch_src = [torch.LongTensor([SOS_ID]+src+[EOS_ID]) for src in batch_src_token]
    src_x = pad_sequence(batch_src, True, PAD_ID)
    src_mask = get_padding_mask(src_x, PAD_ID)
    
    prob_sent = batch_greedy_decode(model, src_x, src_mask)
    print(prob_sent)
