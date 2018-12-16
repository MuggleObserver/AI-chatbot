import numpy as np
import tensorflow as tf

SRC_TRAIN_DATA = "index20wtrain.post"  # 问题输入语言
TRG_TRAIN_DATA = "index20wtrain.resp"  # 答案输入语言
SRC_TRAIN_DATA_NEG = "index20wtrainwcy.post"  # 问题输入语言
TRG_TRAIN_DATA_NEG = "index20wtrainwcy.resp"  # 答案输入语言

CHECKPOINT_PATH = "./IR_ckpt"

HIDDEN_SIZE = 300  # LSTM的隐藏层规模
NUM_LAYERS = 2  # 深层循环神经网络中的LSTM结构的层数
VOCAB_SIZE = 10000  # 词汇表的大小（词汇表按照词频，由高到低向下排列）
BATCH_SIZE = 100  # 训练数据batch的大小
NUM_EPOCH = 2  # 使用训练数据的轮数
KEEP_PROB = 0.8  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限
BASE_LEARNING_RATE = 0.1  # 初始的学习率
LEARNING_RATE_DECAY = 0.99  # 学习衰减率

MAX_LEN = 50  # 限定句子的最大单词数量。

global_step = tf.Variable(0, trainable=False)


def getminbed():
    f = open('minbed', 'r')
    embedict = {}
    rawlist = []
    wordindex = {}
    for i in range(9863):
        p = f.readline()
        p = p.strip()
        list = p.split()
        key = list.pop(0)
        list2 = []
        for t in list:
            list2.append(float(t))
        embedict.update({key: list2})
        rawlist.append(list2)
    f.close()
    embedlist = np.array(rawlist)
    wordindex = {}
    i = 0
    for key in embedict.keys():
        wordindex[key] = i
        i = i + 1
    return rawlist


# 定义IRModel类来描述模型
class IRModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
             for _ in range(NUM_LAYERS)])

        init = tf.constant_initializer(getminbed())

        # 为源语言和目标语言分别定义词向量
        self.embedding = tf.get_variable("src_emb", [VOCAB_SIZE, HIDDEN_SIZE], initializer=init, trainable=False)
        self.W = tf.get_variable("weights", [HIDDEN_SIZE, HIDDEN_SIZE],
                                 initializer=tf.truncated_normal_initializer())

    # 在forward函数中定义模型的前向计算图。
    # src_input,src_size,trg_input,trg_size,targets分别是上面MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_size, targets):
        batch_size = tf.shape(src_input)[0]
        # src_emb和trg_emb的维度均为batch_size * max_time * HIDDEN_SIZE
        src_emb = tf.nn.embedding_lookup(self.embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.embedding, trg_input)

        # 使用dynamic_rnn构造编码器。
        with tf.variable_scope("encoder"):
            enc_src_outputs, enc_src_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, sequence_length=src_size,
                                                               dtype=tf.float32)
            enc_trg_outputs, enc_trg_state = tf.nn.dynamic_rnn(self.enc_cell, trg_emb, sequence_length=trg_size,
                                                               dtype=tf.float32)
            # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类的tuple，enc_state存储的是每一层的最后一个step的输出
            # enc_src_outputs存储的是顶层LSTM的每一步输出，它的维度为[batch_size,max_time,HIDDEN_LAYER]
            # src_enc_output, trg_enc_output的shape为BATCH_SIZE * HIDDEN_SIZE
            src_enc_output = enc_src_outputs[:, -1, :]
            trg_enc_output = enc_trg_outputs[:, -1, :]
        # 通过编码生成的src_enc_output与W相乘，生成可能的回复generate_trg。
        # 通过点乘的方式来预测生成的回复generate_trg和候选的回复trg_enc_output之间的相似程度，
        # 点乘结果越大表示候选回复作为回复的可信度越高；
        # 之后通过sigmoid函数归一化，转成概率形式。
        generate_trg = tf.matmul(src_enc_output, self.W)
        logits = tf.reduce_sum(tf.multiply(generate_trg, trg_enc_output), axis=1, keepdims=True)
        probs = tf.sigmoid(logits)

        # 计算 the binary cross-entropy loss
        targets = tf.expand_dims(targets, 1)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=probs, labels=tf.to_float(targets))
        # Mean loss across the batch of examples
        mean_loss = tf.reduce_mean(losses, name="mean_loss")
        learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,
            global_step=global_step,
            decay_steps=100,
            decay_rate=LEARNING_RATE_DECAY
        )
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mean_loss, global_step=global_step)
        return mean_loss, train_step


def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数。
    dataset = dataset.map(
        lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    dataset = dataset.map(lambda x: (x, tf.size(x)))

    return dataset


def MakeSrcTrgDataset(src_path, trg_path, src_path_neg, trg_path_neg, batch_size):
    # 统计src_path所含的总行数（src_path与trg_path所含行数相等）
    def countLine(filepath):
        count = 0
        for index, line in enumerate(open(filepath, 'r')):
            count += 1
        return count

    line_num = countLine(src_path)

    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    src_data_neg = MakeDataset(src_path_neg)
    trg_data_neg = MakeDataset(trg_path_neg)

    data_label_1 = tf.data.Dataset.from_tensor_slices(tf.ones([line_num], dtype=tf.int32))
    data_label_0 = tf.data.Dataset.from_tensor_slices(tf.zeros([line_num], dtype=tf.int32))

    # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds
    # 由4个张量组成：
    #   ds[0][0]是源句子
    #   ds[0][1]是源句子长度
    #   ds[1][0]是目标句子
    #   ds[1][1]是目标句子长度
    #   ds[2][0]是标签
    dataset_1 = tf.data.Dataset.zip((src_data, trg_data, data_label_1))
    dataset_0 = tf.data.Dataset.zip((src_data_neg, trg_data_neg, data_label_0))
    dataset = dataset_1.concatenate(dataset_0)

    # 删除内容为空（只包含<EOS>）的句子和长度过长的句子。
    def FilterLength(src_tuple, trg_tuple, label):
        ((src_input, src_len), (trg_label, trg_len), label) = (src_tuple, trg_tuple, label)
        src_len_ok = tf.logical_and(
            tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(
            tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(FilterLength)

    # 随机打乱训练数据。
    dataset = dataset.shuffle(400000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子是长度未知的向量
         tf.TensorShape([])),  # 目标句子长度是单个数字
        tf.TensorShape([]))  # 标签是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch
    # 重复训练步骤，直至遍历完Dataset中所有的数据
    while True:
        try:
            # 运行train_op，并计算损失值。训练数据在main()函数中以Dataset方式提供。
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" % (step, cost))
            # 每200步保存一个checkpoint。
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    initializer = tf.random_uniform_initializer(- 0.25, 0.25)

    with tf.variable_scope("IRModel", reuse=None, initializer=initializer):
        train_model = IRModel()
    # 定义输入数据。
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, SRC_TRAIN_DATA_NEG, TRG_TRAIN_DATA_NEG, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_size), label = iterator.get_next()
    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_size, label)

    # 训练模型。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == "__main__":
    main()
