'''
朴素贝叶斯分类器：过滤恶意留言
'''
import numpy as np
import re
# 句子切词规则
def get_tokens(document, stop_list):
    tokens = []
    [tokens.append(token) for token in re.findall('[a-zA-Z]+', document.lower()) if token not in stop_list
        and len(token) > 3]
    return tokens
# 获得文本集合的词条列表和分类列表
def load_dataSet():
    documents = [
        'My dog has flea problems, help please',
        'Maybe not take him to dog park, stupid',
        'My dalmation is so cute, I love him',
        'Stop posting stupid worthless garbage',
        'Mr licks ate my steak, how to stop him',
        'Quit buying worthless dog food, stupid',
    ]  # 总留言表
    tokens_list = []  # 词条列表
    [tokens_list.append(get_tokens(document=document, stop_list=stop_list)) for document in documents]
    classes_list = [0, 1, 0, 1, 0, 1]  # 类别表，1代表侮辱性评论，0代表正常评论
    return tokens_list, classes_list
# 构造词汇表
def create_vocab_list(tokens_list):
    vocab_set = set([])
    for tokens in tokens_list:
        vocab_set = vocab_set | set(tokens)  # 求两个set集合的并集
    return list(vocab_set)
# 根据词汇表将词条列表转化为词条向量：词集模型
def setof_words_to_vec(vocab_list, tokens):
    document_vec = np.zeros(len(vocab_list))
    for token in tokens:
        # 特征值定义：词汇表中的单词是否出现在当前句子中
        if token in vocab_list:
            document_vec[vocab_list.index(token)] = 1  # 1表示单词出现在当前句子中
    return document_vec
# 基于词向量训练朴素贝叶斯分类器——计算目标概率映射：边缘概率P(Ci)、条件概率P(W|Ci)
def train_NB(train_tokens_vec_matrix, train_classes, num_classes_values=2):
    num_vecs_matrix = len(train_tokens_vec_matrix)  # 词向量的样本总数目
    num_vec_words = len(train_tokens_vec_matrix[0])  # 每个词向量的特征值数目
    count_p_w_and_c_arr = []  # 分子数组，统计每个词向量的P(W∩Ci)
    [count_p_w_and_c_arr.append(np.ones(num_vec_words)) for _ in range(num_classes_values)]  # 初始化P(W∩c)
    count_p_c_arr = [2.0] * num_classes_values  # 分母数组，统计每个词向量的P(Ci)
    for i in range(num_vecs_matrix):  # 阅读所有词向量
        Wi = train_tokens_vec_matrix[i]  # 当前词向量Wi = {w1,w2,...,wn}
        Ci = train_classes[i]  # 当前Wi对应的分类Ci
        count_p_w_and_c_arr[Ci] += Wi  # 统计P(Wi∩Ci)：for w in Wi中w∩Ci共同发生的次数
        count_p_c_arr[Ci] += np.sum(Wi)  # 统计P(Ci)：Wi中Ci发生的次数
    # 计算条件概率的极大似然估计：P(Wi|Ci) = P(Wi∩Ci) / P(Ci)
    p_w_or_c_fx_arr = []
    [p_w_or_c_fx_arr.append(np.log(p_w_and_c / p_c)) for p_w_and_c, p_c in zip(count_p_w_and_c_arr, count_p_c_arr)]
    p_class1 = np.where(np.array(train_classes) == 1)[0].shape[0] / num_vecs_matrix  # 单独计算P(C1) = C1 / N
    return p_w_or_c_fx_arr, [1.0 - p_class1, p_class1]
# 用新样本文本测试朴素贝叶斯分类器
def exam_NB(vocab_list, p_w_or_c_fx_arr, p_c_arr):
    test_documents = ['Love my dalmation', 'Stupid garbage']  # 新样本文本
    for i in range(len(test_documents)):
        tokens = get_tokens(document=test_documents[i], stop_list=stop_list)  # 词条列表
        tokens_vec = setof_words_to_vec(vocab_list=vocab_list, tokens=tokens)  # 词条向量
        classify = classify_NB(tokens_vec=tokens_vec, p_w_or_c_fx_arr=p_w_or_c_fx_arr, p_c_arr=p_c_arr)
        print('测试句子：{} 分类结果为：{}'.format(test_documents[i], '侮辱类' if classify == 1 else '非侮辱类'))
# 通过朴素贝叶斯分类器计算的概率，推理新词向量的分类结果
def classify_NB(tokens_vec, p_w_or_c_fx_arr, p_c_arr):
    # 1、基于条件独立性假设，计算似然概率：P(Wi|Ci) = P(w1|Ci)P(w2|Ci)P(w3|Ci)...P(wn|Ci)
    # 2、计算后验概率：P(Ci|Wi) = P(Wi|Ci) * P(Ci)
    classifies = []
    [classifies.append(np.sum(tokens_vec * p_w_or_c) + np.log(p_c)) for p_w_or_c, p_c in zip(p_w_or_c_fx_arr, p_c_arr)]
    return np.argmax(classifies)
if __name__ == '__main__':
    stop_list = []  # 停词表
    with open('./tables/stoplist.csv', 'r', encoding='utf-8') as stoplist_csv:
        [stop_list.append(line.strip('\n')) for line in stoplist_csv.readlines()]
    # 根据样本空间，获得词条列表和分类列表
    tokens_list, classes_list = load_dataSet()
    # 根据词条列表获得不重复的词汇表（词汇表是用来构建特征词条向量）
    vocab_list = create_vocab_list(tokens_list=tokens_list)
    train_tokens_vec_matrix = []  # 训练样本的词条向量数组
    # 将词条列表转化为词条向量列表：1个文档对象 = 1组词条 = 1组词条向量
    for tokens in tokens_list:
        tokens_vec = setof_words_to_vec(vocab_list=vocab_list, tokens=tokens)  # 词汇表作为翻译标准，将词条列表翻译为词条向量
        train_tokens_vec_matrix.append(tokens_vec)
    # 训练朴素贝叶斯分类器
    p_w_or_c_fx_arr, p_c_arr = train_NB(train_tokens_vec_matrix=train_tokens_vec_matrix, train_classes=classes_list)
    # 测试朴素贝叶斯分类器
    exam_NB(vocab_list=vocab_list, p_w_or_c_fx_arr=p_w_or_c_fx_arr, p_c_arr=p_c_arr)
