import tensorflow as tf
import numpy as np
import time
import linecache

CHECKPOINT_PATH = "./IR_ckpt-5200"
TRG_TRAIN_DATA = "index20wtrain.resp"  # 答案输入语言
TRG_TRAIN_DATA_TXT = "index20wtrain.txt"  # 答案输入语言
ENC_TRG_DATASET = "enc_trg_dataset.txt" # 候选答案的编码数据集
TRG_TRAIN_DATA_SIZE = 100    # 候选答案集的大小
HIDDEN_SIZE = 300           # LSTM的隐藏层规模
NUM_LAYERS = 2              # 深层循环神经网络中的LSTM结构的层数
VOCAB_SIZE = 10000          # 词汇表的大小（词汇表按照词频，由高到低向下排列）
MAX_LEN = 50                # 限定句子的最大单词数量。



# 定义IRModel类来描述模型
class IRModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
         for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量
        self.embedding = tf.get_variable("src_emb", [VOCAB_SIZE, HIDDEN_SIZE])
        self.W = tf.get_variable("weights", [HIDDEN_SIZE, HIDDEN_SIZE],
                                 initializer=tf.truncated_normal_initializer())

    # encode_Q的作用为将输入的一条问题进行编码成shape = [1,HIDDEN_SIZE] 的向量
    def encode_Q(self, src_input):

        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)

        src_emb = tf.nn.embedding_lookup(self.embedding, src_input)

        # 使用dynamic_rnn构造编码器。
        with tf.variable_scope("encoder"):
            enc_src_outputs, enc_src_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, sequence_length=src_size,
                                                               dtype=tf.float32)
            # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类的tuple，enc_state存储的是每一层的最后一个step的输出
            # src_enc_output的shape为BATCH_SIZE * HIDDEN_SIZE
            src_enc_output = enc_src_outputs[:, -1, :]

        return src_enc_output


def main():
    # 定义训练用的循环神经网络
    # 定义一个测试例子。
    s = time.time()
    dataset = np.loadtxt(ENC_TRG_DATASET)
    e = time.time()
    print('dataset=', dataset.shape)
    print(e-s)
    test_sentence_Q = [10, 128, 678, 4, 2]

    with tf.variable_scope("IRModel", reuse=None):
        model = IRModel()

    src_enc_output = model.encode_Q(test_sentence_Q)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)
    Q_output = sess.run(src_enc_output)
    sess.close()
    def sigmoid(x):
        s = 1 / (1 + np.exp(x))
        return s
    # 该函数用numpy，会快
    # Q(shape = [1,300])与 A（shape = [num , 300]）按位相乘得到rank（shape = [num,]）,
    # 排序比大小，返回最大的答复及其分数（分数=sigmoid（分数））
    def getMaxScore(Q, R):
        QR = np.dot(Q, R.T)  # 问题和答案矩阵相乘
        maxscore = QR.max()  # QR中的最大值
        maxscore = sigmoid(maxscore) #sigmoid最大值
        maxposition = QR.argmax(axis=1)  # QR中最大值所在位置，也是R中对应句子的行数

        def getMarvelIndex(filename, row):
            words = linecache.getline(filename, row)
            words = words.strip('\n')
            wordslist = words.split(' ')
            for i in range(0, len(wordslist)):
                wordslist[i] = int(wordslist[i])
            return wordslist

        templist = getMarvelIndex(TRG_TRAIN_DATA_TXT, int(maxposition + 1))  # 将分数最大的那个句子提取成列表

        return maxscore, templist


    s = time.time()
    score, resp = getMaxScore(Q_output, dataset)
    e = time.time()
    print(e-s)

    print(score)
    print(resp)

if __name__ == "__main__":
    main()