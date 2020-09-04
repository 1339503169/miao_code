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
model_path =''
dict_path = ''

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
                        supervisor_contract=[]

                        contract_type = []
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
