#coding:utf-8
import time
start=time.time()
import re
import codecs
from keras_bert import Tokenizer,load_trained_model_from_checkpoint
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras_bert import get_custom_objects
from keras.models import load_model,Model
from keras.optimizers import Adam
import numpy as np
# from dataprocess import get_spe_data

# 读取训练集和测试集
# model_path=''
model_path=''
dict_path=''
config_path = ''
checkpoint_path = ''


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    # x = layers.Bidirectional(LSTM(units=256, return_sequences=False), merge_mode='sum')(x)
    x = Dense(200, activation='relu')(x)
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),  # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model
def get_result(list,dict_path,model_path,maxlen=500):
    """
    :param list: 合同文本内容的列表  不需要分词
    :param dict_path: 中文bert预训练模型的词汇表位置
    :param model_path: 训练好的模型文件的位置
    :param maxlen: 输入bert模型的长度  不超过512
    :return:返回合同类型名称
    """
    maxlen = maxlen  # 设置序列长度为120，要保证序列长度不超过512
    # 预训练好的模型
    # 词汇表位置
    dict_path = dict_path
    # 将词表中的词编号转换为字典
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    # 重写tokenizer
    class OurTokenizer(Tokenizer):
        def _tokenize(self, text):
            R = []
            for c in text:
                if c in self._token_dict:
                    R.append(c)
                elif self._is_space(c):
                    R.append('[unused1]')  # 用[unused1]来表示空格类字符
                else:
                    R.append('[UNK]')  # 不在列表的字符用[UNK]表示
            return R

    tokenizer = OurTokenizer(token_dict)

    # 让每条文本的长度相同，用0填充
    def seq_padding(X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])

    # 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
    def acc_top2(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top2_list(logits):
        # 获取列表最大两个值的索引
        top1 = np.argmax(logits)
        new_logits=np.delete(logits,np.argmax(logits))
        top2 = np.argmax(new_logits)
        if top1 > top2:
            return [top1, top2]
        else:
            return [top1, top2 +1]

    custon_objects = get_custom_objects()
    custon_objects['acc_top2'] = acc_top2
    # model=build_bert(343)
    # model.load_weights(model_path)
    model=load_model(model_path,custon_objects)
    def get_test_data(i):
        # 获取符合模型的训练数据
        X1 = []
        X2 = []
        i = i[:maxlen]
        x1, x2 = tokenizer.encode(first=i)
        X1.append(x1)
        X2.append(x2)
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        return [X1, X2]

    print(get_test_data(list))
    test_model_pred = model.predict(get_test_data(list))
    test_pred_top = top2_list(test_model_pred)
#   合同到编号的字典，便于查找原合同
    contract2int={}
    int2contract = {value: key for key, value in contract2int.items()}
    top2_result = [re.sub(r"[^\u4e00-\u9fa5]", " ", int2contract[i]) for i in test_pred_top]

    return top2_result

import docx
doc=docx.Document(file)
text=''
for p in doc.paragraphs:
    text=text+p.text
# text1='重庆市非成套公有住房出售 购买 合同重庆市非成套公有住房出售 购买 合同合同编号 mark重庆市非成套公有住房出售 购买 合同出售方 以下简称甲方  mark购买方 以下简称甲方  mark根据 重庆市非成套公有房屋出售管理办法 及相关房改政策的规定 甲 乙双方协商一致 签订如下协议乙方自愿向甲方购买下表所列住房房屋坐落 地区类别建筑结构 间 数建筑面积 其中分摊面 积配套情况甲方同意乙方按以下第mark种方式购买乙方在购房面积控制标准之内的面积 超面积mark平方米按该房地区类别成本价mark元 平方采购买 超面积mark平方米按该房地区的市场价mark元 平方采购买以经评估的价格mark元 平方米 并享受工龄折扣购买 其中年工龄mark年以经评估的价格mark元 平方采购买 不享受工龄折扣以该住房所在地区类别成本价的18 购买乙方按以上方式购买该住房应付房价款mark元 大写 mark拾mark万mark仟mark佰mark拾mark元mark角mark分  此款由乙方在mark年mark月mark日一次性付给甲方 甲方于乙方缴款次月起停收住房租金 甲方提取房价款的30 计mark元建立住房公共维修基金 其余部分金额mark元为甲方的售房收入 住房公共维修基金及售房收入均应由甲方专项交存住房资金管理机构在银行开设的专户乙方购买该房后拥有完全产权 按照 重庆市房改住房再交易管理办法 的规定可依法进入市场交易 在补交土地使用权出让金和按规定缴纳税费后 收入归乙方所有乙方所购住房户内及自用设施的维修由乙方自行负责 其共用部位及共用设施的维修按有关规定执行甲方负责统一办理 房屋所有权证 和 国有土地使用证  办好后即交乙方执存 有关办证的费用由甲 乙方双方按政策规定各自承担其它本合同一式四份 甲 乙方签字盖章并由乙方在合同约定时间交清房价款后生效 本合同甲 乙双方各执一份 并交房地产权属登记机关两份甲方 mark 签章  乙方 mark 签字法定代表人 mark 签章  代理人 mark 签字mark年mark月mark日 mark年mark月mark日'
print(get_result(text,model_path=model_path,dict_path=dict_path,maxlen=256))
