import os
import sys

from antlr4.error.ErrorListener import ErrorListener
from antlr4 import *
from antlr4_impl.JavaLexer import JavaLexer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
except:
    from .Exceptions import KnownException


class SilentErrorListener(ErrorListener):
    def __init__(self):
        super().__init__()

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        # 静默处理错误，不输出任何内容
        pass


silent_listener = SilentErrorListener()


class JavaCodeExtractor:
    comment_tokens = {
        JavaLexer.COMMENT,
        JavaLexer.LINE_COMMENT
    }

    def __init__(self, code):
        # 预处理：去除 import 行
        code = "".join([t for t in code.splitlines(keepends=True) if not t.startswith("import")]).strip()

        # 检查是否为类定义开头
        t = code[:24]
        if " class " in t or " interface " in t or "import " in t:
            raise KnownException("Cannot extract features from class definition")
        del t

        # 包裹成完整类结构以便解析
        self.code = f"class A{{{code}}}"
        self.input_stream = InputStream(self.code)
        self.lexer = JavaLexer(self.input_stream)

        # 移除默认错误监听器
        self.lexer.removeErrorListeners()
        self.tokens = CommonTokenStream(self.lexer)
        self.tokens.fill()

        # 提取非注释代码 token 及其原始上下文
        self.code_seq = ""
        input_str = self.input_stream.getText(0, self.input_stream.size)  # 获取整个输入字符串

        prev_index = 0
        for token in self.tokens.tokens:
            if token.type in JavaCodeExtractor.comment_tokens:
                continue
            if token.type == Token.EOF:  # 跳过 EOF token
                continue
            start_idx = token.start
            end_idx = token.stop + 1  # 包含 stop 索引

            # 提取两个 token 之间的内容（包含换行）
            segment = input_str[prev_index:start_idx]
            self.code_seq += segment

            # 添加当前 token 内容
            self.code_seq += token.text

            prev_index = end_idx

        # 添加最后一个 token 到结尾的内容
        self.code_seq += input_str[prev_index:]
        self.code_seq = self.code_seq[8:-2].strip()

    def get_codes(self):
        return self.code_seq