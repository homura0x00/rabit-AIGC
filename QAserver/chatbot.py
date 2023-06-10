from question_classifier import gossip_robot, medical_robot, classifier  # 问答模块
import gradio as gr     # 展示界面模块


class Chat:

    def chat_main(self, sent):
        user_sent = classifier(sent)
        print(user_sent)
        if user_sent in ["greet", "goodbye", "deny", "isbot"]:
            reply = gossip_robot(user_sent)
        elif user_sent == "accept":
            reply = "hi，机器人小华很高心为您服务"
        else:
            reply = medical_robot(sent)

        return reply


def chatUI(message, history):
    history = history or []     # 对话消息历史
    qa_handler = Chat()      # 调用问答系统接口
    qa_sent = message           # 输入
    Answers = qa_handler.chat_main(qa_sent)     # 结果的输出
    history.append((message, Answers))      # 将问句和答句保存
    return history, history     # 输出和对话状态参数


chatbot = gr.Chatbot().style(color_map=("green", "pink"))
iface = gr.Interface(
    fn=chatUI,
    inputs=["text", "state"],       # 用户输入窗口
    outputs=[chatbot, "state"],     # 问题反馈和对话显示窗口
    title="问答系统界面",              # 交互界面的名称
    allow_flagging="never",         # 历史保存按钮，never值指不显示
)

iface.launch()      # 启动交互界面
