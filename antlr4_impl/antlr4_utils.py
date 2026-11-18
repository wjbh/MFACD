class __Antlr4Utils:
    @staticmethod
    def get_raw_text(node, input_stream):
        start_idx = node.start.start  # 起始字符索引
        stop_idx = node.stop.stop  # 结束字符索引
        return input_stream.strdata[start_idx:stop_idx + 1]


Antlr4Utils = __Antlr4Utils()
