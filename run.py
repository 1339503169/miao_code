# -*- coding: UTF-8 -*-
from flask import Flask, Response, request
import json, codecs
from keras_bert import Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras_bert import get_custom_objects
from keras.models import load_model, Model
import tensorflow as tf
import numpy as np
import re

app = Flask(__name__)
model = None

# 首先载入模型
model_path = './bert/bert_dumpnew0.hdf5'
dict_path = './bert/vocab.txt'

maxlen = 64
token_dict = {}
with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# # 重写tokenizer
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


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def get_test_data(text):
    # 获取符合模型的训练数据
    X1 = []
    X2 = []
    text = text[:maxlen]
    x1, x2 = tokenizer.encode(first=text)
    X1.append(x1)
    X2.append(x2)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    return [X1, X2]


def top2_list(logits):
    # 获取列表最大两个值的索引
    top1 = np.argmax(logits)
    new_logits = np.delete(logits, np.argmax(logits))
    top2 = np.argmax(new_logits)
    if top1 > top2:
        return [top1, top2]
    else:
        return [top1, top2 + 1]


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def load_bert_model():
    custon_objects = get_custom_objects()
    custon_objects['acc_top2'] = acc_top2
    model = load_model(model_path, custon_objects)
    return model
graph = tf.get_default_graph()
model = load_bert_model()
import time
@app.route('/contract', methods=['POST', 'GET'])
def contract():
    start_time=time.time()
    if request.method == 'GET':
        # 如果请求方式是get 返回一个测试的句子
        return Response(json.dumps([{"200": "method wrong"}], ensure_ascii=False))
    elif request.method == 'POST':

        # 如果请求方式是post 则进入下一个判断
        if request.values.get('bargainText'):

            html = request.values.get('bargainText')

            # html = risks.html_to_plain_text(html)
            # 判断合同文本是否存在 如果存在 进入下一个判断
            if request.values.get('type'):

                # 判断合同类型是否传入 如果传入 进入判断合同风险点模块
                if len(html.strip())<20:
                    # 传入文本为空 直接返回
                    end_time=time.time()
                    Response(json.dumps([{"250": "bargainText is null"}], ensure_ascii=False))
                else:

                    html = html.replace('\u3000', '')
                    # 判断合同文本中是否存在禁止交易的词汇
                    from dataProcess.objExam import Jieba
                    jie = Jieba()
                    if jie.find_forbbiden_object(html):
                        end_time = time.time()
                        return Response(json.dumps([{"100": "forbidden object"}], ensure_ascii=False))
                    else:

                        # 如果合同文本不存在不能交易的物品 进入判断风险点阶段
                        type1 = request.values.get('type')
                        supervisor_contract=['光船租赁合同','其他海事海商类合同','定期租船合同','海上保险合同','海上运输合同',
                                             '海员劳务合同','港口作业合同','海难救助合同','航次租赁合同','船舶买卖合同','船舶建造合同'
                                             ,'船舶租赁合同','中外合作经营企业合同','中外合资经营企业合同','中外定期租船合同','中外航次租船合同',
                                             '其他涉外类合同','国际保险合同','国际建设工程合同','国际成套设备项目合同','国际租赁合同','国际货物运输贸易合同',
                                             '国际货物运输合同','国际贷款合同','国际贸易代理合同','多式联运合同','对外加工装配合同','对外劳务合作合同',
                                             '补偿贸易合同','彩票服务合同','网络空间租赁合同','汽车租赁合同','船舶租赁合同','融资租赁合同','设备租赁合同',
                                             '其他投资理财类合同','外汇交易合同','委托理财合同','期货交易合同','证券上市合同','证券交易合同','证券回购合同',
                                             '证券承销合同','证券投资咨询合同','其他身份关系类协议','抚养协议','收养协议','离婚协议','财产协议','遗产协议',
                                             '信托合同','储蓄存款合同','其他银行业务类合同','开户合同','网上银行合同','银行结算合同','个人养老金保险条款',
                                             '人寿保险合同','企业财产保险合同','住房保险合同','保险经纪合同','健康保险合同','其他保险类合同','医疗保险合同',
                                             '家庭财产保险合同','建筑工程保险合同','意外伤害保险合同','财产保险合同','责任保险合同','货物运输保险合同',
                                             '运输工具保险合同','其他信托类合同','基金信托合同','股权信托合同','财产信托合同','资产信托合同','专向资金贷款合同',
                                             '买方信贷、政府贷款和混合借贷合同','住房贷款合同','保证担保借款合同','信托贷款合同','信用贷款合同','借款展期合同',
                                             '农业贷款合同','助学贷款协议','固定资产贷款合同','国际借款合同','外汇借款合同','对外承包项目借款合同','工程建设贷款合同',
                                             '技术改造借款合同','民间借贷合同','汽车贷款合同','消费借款合同','联营股本贷款合同','转贷款协议','配套人民币贷款合同',
                                             '银团借款合同','土地出让合同','招投标买卖合同','政府采购合同','船舶买卖合同','调拨合同','进出口买卖合同','中外合作经营合同',
                                             '中外合资经营合同','公司章程','房产开发合同','改制合同','联营合同','土地征用合同','房屋拆迁合同','补偿安置合同'
                                ]

                        contract_type = ['房屋买卖合同', '劳动合同', '一般买卖合同', '房屋租赁合同', '装饰装修合同',
                                         '工程设计合同',  '加工合同', '专利转让合同', '抵押合同',
                                         '专利实施许可合同', '技术转让合同', '商标转让合同', '维修合同', '商标许可合同', '技术许可合同',
                                          '版权许可合同', '版权转让合同', '场地租赁合同', '工程咨询合同', '采购合同',
                                         '技术服务合同', '工矿产品买卖合同', '软件著作权转让合同', '软件著作权许可使用合同', '软件开发合同',
                                         '土地租赁合同','质押合同','经销合同']
                        if type1 in supervisor_contract:
                            return Response(json.dumps([{"600": "strong regulatory contract"}], ensure_ascii=False))
                        if type1 not in contract_type:
                            return Response(json.dumps([{"300": "unsupported contract type"}], ensure_ascii=False))
                        # 按 。 切分文本 此处有待于改进
                        from risks import risks
                        ridk = risks(type1)
                        # html1 = html.split('。')
                        # html = re.sub('。。', '。', html)
                        from contract_risks.identification import Id_process
                        res = ridk.process(html)
                        # try:
                        #
                        # except Exception as err:
                        #     print(err)
                        #     return Response(json.dumps([{'500':'contract form is not reasonable'}],ensure_ascii=False))
                        if type(res)==dict:
                            return Response(json.dumps([{'500':'contract form is not reasonable'}],ensure_ascii=False))
                        try:
                            iden = Id_process(html, type1)
                        except Exception as e:
                            print(e)
                            return Response(json.dumps([{'500':'contract form is not reasonable'}],ensure_ascii=False))
                        ress={}
                        for key,value in res[1].items():
                            if value==0:
                                ress[key]=value
                        result = (iden, res[0], ress)
                        # try:
                        #     res = ridk.process(html)
                        #     iden = Id_process(html, type1)
                        #     result = (iden, res[0], res[1])
                        # except Exception as err:
                        #     end_time = time.time()
                        #     print('用时{}'.format(end_time - start_time))
                        #     return Response(
                        #         json.dumps({'500': "Problems with identity recognition excepition is{}".format(err)},
                        #                    ensure_ascii=False))
                        end_time = time.time()
                        print('用时{}'.format(end_time - start_time))
                        print(json.dumps(result, ensure_ascii=False))
                        return Response(json.dumps(result, ensure_ascii=False))
            else:
                # 如果合同类型不存在 则进入判断合同文本类型模块
                html = request.values.get('bargainText')
                if len(html.strip())<20:
                    # 合同文本为空的话 报错
                    end_time = time.time()
                    print('用时{}'.format(end_time - start_time))
                    Response(json.dumps([{"250": "bargainText is null"}], ensure_ascii=False))
                else:
                    from type_main import Contract
                    con = Contract()
                    html = ' '.join(con.html_to_plain_text(html))
                    # 如果合同文本不存在不能交易的物品 进入判断合同类型阶段
                    from dataProcess.objExam import Jieba
                    jie = Jieba()
                    # from utils.fine_prohibition import find_forbbiden_object
                    if jie.find_forbbiden_object(html):
                        end_time = time.time()
                        return Response(json.dumps([{"100": "forbidden object"}], ensure_ascii=False))
                    else:
                        from contract_risks.identification import classification
                        if classification(html) != 0:
                            print(json.dumps(classification(html), ensure_ascii=False))
                            return Response(json.dumps(classification(html), ensure_ascii=False))
                        else:
                            global graph
                            with graph.as_default():
                                # global model
                                test_pred = model.predict(get_test_data(html))
                                result = top2_list(test_pred)

                            # 该类别之后重新修改
                            with open('./data/contract2number.json', 'r', encoding='utf-8') as f:
                                contract2int = json.load(f)
                            f.close()
                            int2contract = {value: key for key, value in contract2int.items()}
                            top2_result = [int2contract[i] for i in result]
                            typename = {}
                            typename['templateTypeName1'] = top2_result[0]
                            typename['templateTypeName2'] = top2_result[1]
                            end_time = time.time()
                            print('用时{}'.format(end_time - start_time))
                            print(json.dumps(typename, ensure_ascii=False))
                            return Response(json.dumps(typename, ensure_ascii=False))
        else:
            # 合同文本如果不存在 则返回报错信息
            return Response(json.dumps([{"240": "bargainText is null"}], ensure_ascii=False))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)
