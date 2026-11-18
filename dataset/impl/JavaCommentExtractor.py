import os
import re
import sys

from antlr4.error.ErrorListener import ErrorListener
from antlr4_impl.JavaConst import Antlr4JavaConst
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


class JavaCommentExtractor:
    comment_tokens = Antlr4JavaConst.comment_tokens

    def __init__(self, code):
        code = "".join([t for t in code.splitlines(keepends=True) if not t.startswith("import")]).strip()
        t = code[:24]
        if " class " in t or " interface " in t or "import " in t:
            raise KnownException("Cannot extract features from class definition")
        del t
        self.code = f"class A{{\n{code}\n}}"
        self.input_stream = InputStream(self.code)
        self.lexer = JavaLexer(self.input_stream)
        self.lexer.removeErrorListeners()
        self.lexer.addErrorListener(silent_listener)
        self.tokens = CommonTokenStream(self.lexer)
        self.tokens.fill()
        self.comment_seq = []
        if self.tokens:
            for token in self.tokens.tokens:
                if token.channel == Lexer.HIDDEN and token.type in {JavaLexer.COMMENT, JavaLexer.LINE_COMMENT}:
                    comment_text = token.text.strip()
                    self.comment_seq.append(comment_text)
        self.comment_seq = self._clean_comments(self.comment_seq)
        self.comment_seq = "\n".join(self.comment_seq)

    def _clean_comments(self, comments):
        cleaned = []
        for comment in comments:
            if comment.startswith("//"):
                # 单行注释：去掉开头的 //
                cleaned.append(comment[2:].strip())
            elif comment.startswith("/*"):
                # 多行注释：去掉开头 /* 和结尾 */, 并处理中间的 *
                lines = comment.strip("/*").strip("*/").split('\n')
                for line in lines:
                    cleaned_line = re.sub(r'^\s*\*', '', line).strip()  # 去掉开头的 *
                    if cleaned_line:
                        cleaned.append(cleaned_line)
        return cleaned

    def get_comments(self):
        return self.comment_seq
