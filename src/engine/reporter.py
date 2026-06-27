
class Reporter():
    def on_thinking(self) -> None:
        """当模型开始思考时开始调用"""
        pass

    def on_tool_call(self, tool_name: str, args: str) -> None:
        """当模型决定并发执行工具时开始调用"""
        pass
    
    def on_tool_result(self) -> None:
        """当工具在底层执行完毕时调用"""
        pass

    def on_message(self) -> None:
        """当模型任务宣告完成，向用户输出最终纯文本回答时调用"""
        pass