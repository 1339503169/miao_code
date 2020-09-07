import tensorflow as tf
import keras
from keras import layers
import jieba
import re
sorted_vocab=[]
vocab_to_int={}
def clean_str(str):
    # string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", str)
    string = re.sub('(\d+)', '<mark>', str)
    string = re.sub(r"[^\u4e00-\u9fa5<>markRMB]", " ", string)

    return string
def predict(text,textlength,embedding_dimension,model_path):
    text=jieba.lcut(clean_str(text))
    text=[vocab_to_int[i] for i in text]
    # 这个实现的seq_padding的功能
    if len(text)<40:
        text=text+[0]*(40-len(text))
    else:
        text=text[0:40]
    def convolution():
        """
        模型中间层  承接embedding层的结果 进行cnn+lstm特征提取
        :return:
        """
        # 定义输入占位符
        inn = layers.Input(shape=(textlength, embedding_dimension, 1))
        cnn = []
        # 卷积核大小
        filter_size = [2, 4, 6]
        # 对不同大小的卷积核进行操作
        for size in filter_size:
            conv = layers.Conv2D(filters=128, kernel_size=(size, embedding_dimension), strides=8,
                                 padding='valid', activation='relu')(inn)
            # 将数据维度由4维变成3维   作为lstm的输入
            conv = tf.squeeze(conv, axis=2)
            # return_sequences=True 返回每一个时刻的输出   =False 返回最后一个时刻的输出
            lstm1 = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True), merge_mode='sum')(conv)
            lstm2 = layers.Bidirectional(layers.LSTM(units=128, return_sequences=False), merge_mode='sum')(lstm1)
            cnn.append(lstm2)
        # 对不同卷积核LSTM输出的结果进行凭借 作为下一层的输入
        outt = layers.concatenate(cnn)
        # 将输入到输出合成为一个model便于接入Sequential
        model = keras.Model(inn, outt)
        return model

    def CNNLSTM():
        """
        模型主体  主要是定义模型 然后编译
        :return:
        """
        model = keras.Sequential([
            layers.Embedding(input_dim=len(sorted_vocab), output_dim=embedding_dimension),
            layers.Reshape((textlength, embedding_dimension, 1)),
            convolution(),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.05),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['binary_accuracy'])
        return model

    model = CNNLSTM()
    model.load_weights('model_path')
    result=model.predict(text)
    return [result]