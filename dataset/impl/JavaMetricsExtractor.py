import logging
import os
import sys

from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNodeImpl
from antlr4_impl.JavaConst import Antlr4JavaConst
from antlr4_impl.JavaParserListener import JavaParserListener
from antlr4 import *
from antlr4_impl.JavaLexer import JavaLexer
from antlr4_impl.JavaParser import JavaParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
except:
    from .Exceptions import KnownException

class CodeMetricsListener(JavaParserListener):
    operator_tokens = Antlr4JavaConst.operator_tokens  # 假设已定义
    operand_tokens = Antlr4JavaConst.operand_tokens

    def __init__(self):
        self.statement_count = 0  # 语句总数
        self.operator_count = 0  # 运算符总数
        self.cyclomatic_complexity = 1  # 圈复杂度（初始值为1）
        self._decision_points = set()  # 用于记录决策点位置
        self._nesting_stack = []  # 嵌套层级堆栈
        self.total_nesting_depth = 0  # 嵌套深度总和
        self.max_nesting_depth = 0  # 最大嵌套深度
        self.variable_declarations = 0
        self.variable_references = 0
        self.loop_count = 0
        self.expression_count = 0
        self.argument_count = 0
        self.comment_count = 0
        self.comment_line_count = 0
        self.operand_count = 0
        self.branch_count = 0
        self.line_count = 0
        self._last_line = -1
        self.unused_variable_count = 0
        self.variable_reassignment_count = 0
        self.null_check_count = 0
        self.max_line_length = 0  # 最大行长度
        self._current_line_length = {}  # 记录每行的当前长度 {line_number: length}
        self._declared_variables = set()
        self._referenced_variables = set()
        self._variable_assignments = dict()
        self.word_lengths = []

    def get_average_word_length(self):
        if not self.word_lengths:
            return 0.0
        return sum(self.word_lengths) / len(self.word_lengths)

    def visitTerminal(self, node: TerminalNodeImpl):
        line = node.symbol.line
        column = node.symbol.column
        text = node.getText()

        # 计算当前token的结束位置（考虑多行字符串的情况）
        line_span = text.count('\n')  # 计算token跨的行数
        if line_span == 0:
            token_length = column + len(text) - 1
            self._current_line_length[line] = max(
                self._current_line_length.get(line, 0),
                token_length
            )
        else:
            lines = text.split('\n')
            for i, line_text in enumerate(lines):
                current_line = line + i
                if i == 0:
                    line_len = column + len(line_text) - 1
                else:
                    line_len = len(line_text)
                self._current_line_length[current_line] = max(
                    self._current_line_length.get(current_line, 0),
                    line_len
                )
        self.max_line_length= max(self._current_line_length.values())

        # 原有逻辑继续保留
        if node.symbol.type in self.operator_tokens:
            self.operator_count += 1
        if node.symbol.type == JavaLexer.IDENTIFIER and self._is_real_operand(node):
            self.operand_count += 1
        if node.symbol.type not in {JavaLexer.COMMENT, JavaLexer.LINE_COMMENT,
                                    JavaLexer.WS} and node.symbol.line != self._last_line:
            self.line_count += 1
            self._last_line = node.symbol.line
        if node.symbol.type in self.operand_tokens:
            self.operand_count += 1

        if node.symbol.type in {JavaLexer.IDENTIFIER, JavaLexer.STRING_LITERAL}:
            words = self._split_identifier(text)
            for word in words:
                if not word.isdigit():
                    self.word_lengths.append(len(word))
        self._last_line = node.symbol.line

    def _process_comment(self, node):
        """处理注释统计"""
        self.comment_count += 1
        comment_text = node.getText()
        # 计算注释中的行数（按换行符分割）
        lines = comment_text.split('\n')
        self.comment_line_count += len(lines)

        # 调试输出
        # print(f"[Comment] Found at line {node.symbol.line}: {comment_text[:30]}...")

    def _split_identifier(self, text):
        """辅助方法：分割各种编程标识符"""
        if not text:
            return []

        # 处理下划线命名 (snake_case)
        if '_' in text:
            return [w.lower() for w in text.split('_') if w]

        # 处理驼峰命名 (camelCase)，仅当首字母小写时才视为 camelCase
        elif text[0].islower() and any(c.isupper() for c in text):
            return self._split_camel_case(text)

        # 默认作为普通单词或 PascalCase / 全大写处理
        else:
            return [text.lower()]

    def _split_camel_case(self, text):
        """辅助函数：正确分割驼峰命名"""
        words = []
        current_word = [text[0].lower()]
        for i in range(1, len(text)):
            c = text[i]
            prev = text[i - 1]

            # 判断是否是新单词的开始：
            # 1. 当前字符是大写，且前一个不是大写
            # 2. 或者连续大写后遇到小写（如 HTTPRequest -> ['http', 'request']）
            if c.isupper() and not prev.isupper():
                words.append(''.join(current_word))
                current_word = [c.lower()]
            else:
                current_word.append(c.lower())

        words.append(''.join(current_word))
        return [w for w in words if w]  # 过滤空字符串

    def enterMethodDeclaration(self, ctx: JavaParser.MethodDeclarationContext):
        """统计方法参数数量"""
        if ctx.formalParameters().formalParameterList():
            self.argument_count += len(ctx.formalParameters().formalParameterList().formalParameter())

    def enterVariableDeclarator(self, ctx: JavaParser.VariableDeclaratorContext):
        """统计所有变量声明（局部变量、类字段、方法参数等）"""
        self.variable_declarations += 1
        var_name = ctx.variableDeclaratorId().identifier().getText()
        self._declared_variables.add(var_name)
        # 调试用（可选）
        # print(f"[Variable] Declaration at line {ctx.start.line}: {ctx.getText()}")
        if ctx.variableInitializer():
            self._variable_assignments[var_name] = self._variable_assignments.get(var_name, 0) + 1
            # print(f"[Assignment] Initial at line {ctx.start.line}: {var_name}")

    def enterPrimary(self, ctx: JavaParser.PrimaryContext):
        """根据PrimaryContext结构精确识别变量引用"""
        # 情况1：简单变量名 (如 "x")
        if ctx.identifier():
            # 排除方法调用、类字面量等情况
            if not (ctx.LPAREN() or ctx.DOT() and ctx.CLASS()):
                var_name = ctx.identifier().getText()
                self._referenced_variables.add(var_name)
                self.variable_references += 1
                # print(f"[Ref] Variable at {ctx.start.line}:{ctx.start.column} - {ctx.getText()}")

        # 情况2：成员变量访问 (如 "obj.x")
        elif (ctx.DOT() and ctx.identifier() and
              not ctx.CLASS() and not ctx.LPAREN()):
            var_name = ctx.identifier().getText()
            self._referenced_variables.add(var_name)
            self.variable_references += 1
            # print(f"[Ref] Member at {ctx.start.line}:{ctx.start.column} - {ctx.getText()}")

    def enterStatement(self, ctx: JavaParser.StatementContext):
        self.statement_count += 1
        # 检测if语句（包括else）
        if ctx.IF():
            self.branch_count += 1
            self._add_complexity(ctx, "if")
            condition_text = ctx.parExpression().getText()
            if 'null' in condition_text:
                self.null_check_count += 1
                # print(f"[Null Check] Found at line {ctx.start.line}: {condition_text}")
        elif ctx.ELSE():
            self.branch_count += 1

        # 检测for循环
        elif ctx.FOR():
            self.loop_count += 1
            self._add_complexity(ctx, "for")

        # 检测while循环
        elif ctx.WHILE():
            self.loop_count += 1
            self._add_complexity(ctx, "while")

        # 检测do-while循环
        elif ctx.DO():
            self.loop_count += 1
            self._add_complexity(ctx, "do-while")

        # 检测switch中的case（不包括default）
        elif ctx.switchLabel():
            for label in ctx.switchLabel():
                if label.CASE():
                    self.branch_count += 1
                    self._add_complexity(ctx, "case")
                    break

    def enterObjectCreationExpression(self, ctx: JavaParser.ObjectCreationExpressionContext):
        # 遍历所有子节点寻找 IdentifierContext
        var_name = None
        for child in ctx.children:
            if isinstance(child, JavaParser.IdentifierContext):
                var_name = child.getText()
                break

        if var_name and var_name in self._declared_variables:
            self._variable_assignments[var_name] = self._variable_assignments.get(var_name, 0) + 1
            self.variable_reassignment_count += 1
            # print(f"[Reassignment] Found at line {ctx.start.line}: {var_name}")

    def enterLocalVariableDeclaration(self, ctx: JavaParser.LocalVariableDeclarationContext):
        """处理局部变量声明中的多个变量声明（如 int x, y = 1, z;）"""
        # 添加安全检查
        var_decls = ctx.variableDeclarators()
        if var_decls is None:
            return

        declarators = var_decls.variableDeclarator()
        if declarators is None:
            return

        for declarator in declarators:
            var_name = declarator.variableDeclaratorId().identifier().getText()
            self._declared_variables.add(var_name)
            if declarator.variableInitializer():  # 如果有初始化
                self._variable_assignments[var_name] = self._variable_assignments.get(var_name, 0) + 1

    def enterMethodCallExpression(self, ctx: JavaParser.MethodCallExpressionContext):
        for child in ctx.children:
            if isinstance(child, JavaParser.IdentifierContext):
                method_name = child.getText()
                if method_name in {'equals', 'isPresent'}:
                    self.null_check_count += 1
                    # print(f"[Null Check] Found at line {ctx.start.line}: {ctx.getText()}")
                    break

    def enterCatchClause(self, ctx: JavaParser.CatchClauseContext):
        self.branch_count += 1
        self._add_complexity(ctx, "catch")

    def enterBinaryOperatorExpression(self, ctx: JavaParser.BinaryOperatorExpressionContext):
        # 检测逻辑运算符（&&、||）
        if ctx.bop and ctx.bop.text in {'&&', '||'}:
            self.branch_count += 1
            self._add_complexity(ctx, f"logical_{ctx.bop.text}")

    def enterTernaryExpression(self, ctx: JavaParser.TernaryExpressionContext):
        self.branch_count += 1
        self._add_complexity(ctx, "ternary")
        condition_text = ctx.expression(0).getText()
        if 'null' in condition_text:
            self.null_check_count += 1
            # print(f"[Null Check] Found at line {ctx.start.line}: {condition_text}")

    def _add_complexity(self, ctx, node_type):
        """内部方法：增加圈复杂度"""
        location = (ctx.start.line, ctx.start.column)
        if location not in self._decision_points:
            self._decision_points.add(location)
            self.cyclomatic_complexity += 1
            # 调试用（可选）
            # print(f"[Complexity +1] {node_type} at line {ctx.start.line}")

    def enterBlock(self, ctx: JavaParser.BlockContext):
        """进入代码块时增加嵌套层级"""
        if self._nesting_stack:
            current_depth = self._nesting_stack[-1] + 1
        else:
            current_depth = 1

        self._nesting_stack.append(current_depth)
        self.total_nesting_depth += current_depth
        self.max_nesting_depth = max(self.max_nesting_depth, current_depth)

        # 调试用（可选）
        # print(f"[Nesting] Enter block at line {ctx.start.line}, depth={current_depth}")

    def exitBlock(self, ctx: JavaParser.BlockContext):
        """退出代码块时减少嵌套层级"""
        if self._nesting_stack:
            self._nesting_stack.pop()

        # 调试用（可选）
        # print(f"[Nesting] Exit block at line {ctx.start.line}")

    def enterClassBody(self, ctx: JavaParser.ClassBodyContext):
        """类体也算作一层嵌套"""
        self._enter_nested_scope(ctx, "class")

    def exitClassBody(self, ctx: JavaParser.ClassBodyContext):
        self._exit_nested_scope(ctx)

    def enterMethodBody(self, ctx: JavaParser.MethodBodyContext):
        """方法体也算作一层嵌套"""
        self._enter_nested_scope(ctx, "method")

    def exitMethodBody(self, ctx: JavaParser.MethodBodyContext):
        self._exit_nested_scope(ctx)

    def exitCompilationUnit(self, ctx: JavaParser.CompilationUnitContext):
        """在解析结束时计算未使用变量数量"""
        unused_vars = self._declared_variables - self._referenced_variables
        self.unused_variable_count = len(unused_vars)

        # 调试输出（可选）
        # if unused_vars:
        #     print(f"[Unused Variables] Found {self.unused_variable_count} unused variables:")
        #     for var in sorted(unused_vars):
        #         print(f"  - {var}")

    def enterEveryRule(self, ctx: ParserRuleContext):
        """通用规则入口，用于统计表达式"""
        if isinstance(ctx, JavaParser.ExpressionContext):
            self.expression_count += 1
            # 调试输出
            # print(f"[Expression] Found at line {ctx.start.line}: {ctx.getText()}")
            if isinstance(ctx, JavaParser.BinaryOperatorExpressionContext):
                if ctx.bop and ctx.bop.text in {'==', '!='}:
                    left_text = ctx.expression(0).getText()
                    right_text = ctx.expression(1).getText()
                    if 'null' in [left_text, right_text]:
                        self.null_check_count += 1
                        # print(f"[Null Check] Found at line {ctx.start.line}: {ctx.getText()}")

    def _enter_nested_scope(self, ctx, scope_type):
        """通用方法：进入嵌套作用域"""
        if self._nesting_stack:
            current_depth = self._nesting_stack[-1] + 1
        else:
            current_depth = 1

        self._nesting_stack.append(current_depth)
        self.total_nesting_depth += current_depth
        self.max_nesting_depth = max(self.max_nesting_depth, current_depth)

        # 调试用（可选）
        # print(f"[Nesting] Enter {scope_type} at line {ctx.start.line}, depth={current_depth}")

    def _exit_nested_scope(self, ctx):
        """通用方法：退出嵌套作用域"""
        if self._nesting_stack:
            self._nesting_stack.pop()

        # 调试用（可选）
        # print(f"[Nesting] Exit scope at line {ctx.start.line}")

    def _is_real_operand(self, node):
        """更精确的操作数检查"""
        # 排除出现在以下位置的标识符：
        # 1. 方法名位置
        # 2. 类名位置
        # 3. 类型声明位置
        # 通过检查父节点类型实现
        parent = node.parentCtx
        while parent:
            if isinstance(parent, (JavaParser.MethodDeclarationContext,
                                   JavaParser.ClassDeclarationContext,
                                   JavaParser.TypeTypeContext)):
                return False
            parent = parent.parentCtx
        return True





class SilentErrorListener(ErrorListener):
    def __init__(self):
        super().__init__()

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        # 静默处理错误，不输出任何内容
        pass


silent_listener = SilentErrorListener()


class JavaMetricsExtractor:
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
        self.comment_count = 0
        if self.tokens:
            for token in self.tokens.tokens:
                if token.channel == Lexer.HIDDEN and token.type in self.comment_tokens:
                    self.comment_count += 1
        self.parser = JavaParser(self.tokens)
        self.parser.removeErrorListeners()
        self.parser.addErrorListener(silent_listener)
        self.tree = self.parser.compilationUnit()
        self.tree = self.tree.children[0].children[0].children[2].children[1]
        self.listener = CodeMetricsListener()
        walker = ParseTreeWalker()
        walker.walk(self.listener, self.tree)

    def get_number_of_statements(self):
        return self.listener.statement_count

    def get_number_of_operators(self):
        return self.listener.operator_count

    def get_cyclomatic_complexity(self):
        return self.listener.cyclomatic_complexity

    def get_total_depth_of_nesting(self):
        return self.listener.total_nesting_depth

    def get_max_nesting_depth(self):
        return self.listener.max_nesting_depth

    def get_number_of_variable_declarations(self):
        return self.listener.variable_declarations

    def get_number_of_variable_references(self):
        return self.listener.variable_references

    def get_number_of_loops(self):
        return self.listener.loop_count

    def get_number_of_expressions(self):
        return self.listener.expression_count

    def get_number_of_arguments(self):
        return self.listener.argument_count

    # def get_number_of_comments(self):
    #     return self.listener.comment_count
    #
    # def get_number_of_comment_lines(self):
    #     return self.listener.comment_line_count

    def get_number_of_operands(self):
        return self.listener.operand_count

    def get_number_of_branches(self):
        return self.listener.branch_count

    def get_line_of_codes(self):
        return self.listener.line_count

    # def get_comment_density(self):
    #     if self.listener.line_count == 0:
    #         return 0
    #     return self.comment_count / self.listener.line_count

    def get_number_of_unused_variables(self):
        return self.listener.unused_variable_count

    def get_number_of_variable_reassignments(self):
        return self.listener.variable_reassignment_count

    def get_max_line_length(self):
        return self.listener.max_line_length

    def get_average_word_length(self):
        return self.listener.get_average_word_length()

    def get_number_of_null_check(self):
        return self.listener.null_check_count

    def generate_feature_list(self):
        """
        将所有 get_xxx 方法的返回值按顺序组合成一个列表
        """
        feature_list = []
        methods = dir(self)
        # i=0
        for method in methods:
            if method.startswith("get_") and callable(getattr(self, method)) and not method.startswith("get_token"):
                try:
                    value = getattr(self, method)()
                    # print(i, method)
                    feature_list.append(value)
                except Exception as e:
                    # 可选：记录错误或跳过无法调用的方法
                    logging.warning(f"Error calling {method}: {e}")
                    feature_list.append(None)
                # i+=1
        return tuple(feature_list)

    def count_features(self):
        c = 0
        methods = dir(self)
        for method in methods:
            if method.startswith("get_") and callable(getattr(self, method)) and not method.startswith("get_token"):
                c += 1
        return c

    def __str__(self):
        methods = dir(self)
        data = [f"{method[4:].replace('_', ' ')}: {getattr(self, method)()}" for method in methods if
                method.startswith("get") and not method.startswith("get_token")]
        return "\n".join(data)

if __name__ == '__main__':
    extractor = JavaMetricsExtractor("""class A{public void a(){
        // AAAAA
        return 0;
    }}""")
    def ttt(extractor):
        feature_list = []
        name_list = []
        methods = dir(extractor)
        i=0
        for method in methods:
            if method.startswith("get_") and callable(getattr(extractor, method)) and not method.startswith("get_token"):
                try:
                    value = getattr(extractor, method)()
                    print(i, method)
                    name_list.append(method[4:])
                    feature_list.append(value)
                except Exception as e:
                    # 可选：记录错误或跳过无法调用的方法
                    logging.warning(f"Error calling {method}: {e}")
                    feature_list.append(None)
                i+=1
        print(name_list)
    ttt(extractor)