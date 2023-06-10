from Resources.model.clf_model import CLFModel
from medical_classifier import *  # 调用问题分类子函数，对用户所问的问题进行分类
from medical_parser import *   # 调用问题解析子函数，
from answer_search import *  # 查询和反馈用户所需的答案
from config import *    # 闲聊回答模板
import random

clf_model = CLFModel('./Resources/model/model_file')


def classifier(text):
    """
    判断是否是闲聊意图，以及是什么类型闲聊
    """
    return clf_model.predict(text)


def gossip_robot(intent):
    return random.choice(
        gossip_corpus.get(intent)
    )


def medical_robot(text):
    """
    如果确定是诊断意图则使用该方法进行诊断问答
    """
    answer = 'hi~我是AI导诊助手小华医生，小华还在努力提升学习能力，会不断改进完善的，如果有不到之处，还请您多多包含！'
    res_classify = QuestionClassifier().classify(text)  # 读取用户的输入内容，解析
    if not res_classify:
        return answer
    res_sql = QuestionPaser().parser_main(res_classify)
    final_answers = AnswerSearcher().search_main(res_sql)
    if not final_answers:
        return answer
    else:
        return '\n'.join(final_answers)
