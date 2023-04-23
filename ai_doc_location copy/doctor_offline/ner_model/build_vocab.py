import json

# 这里不使用码表原始编码，后续使用BertTokenizer进行编码
def build_vocab():
    chat_to_id = json.load(open('./ner_data/char_to_id.json', mode='r', encoding='utf8'))
    unique_words = list(chat_to_id.keys())[1:-1]
    unique_words.insert(0, '[UNK]')
    unique_words.insert(0, '[PAD]')

    # 将字写入到 data/bilstm_crf_vocab_aidoc.txt 词典文件中
    with open('./ner_data/bilstm_crf_vocab_aidoc.txt', 'w') as file:
        for word in unique_words:
            file.write(word + '\n')

if __name__ == '__main__':
    build_vocab()
