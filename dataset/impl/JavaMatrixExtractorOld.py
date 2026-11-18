import logging
import re

from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNodeImpl
from config import ProjectConfig
from antlr4_impl.JavaConst import Antlr4JavaConst
from antlr4_impl.JavaParserListener import JavaParserListener
from dataset.AbstractFeatureExtractor import AbstractMatrixExtractor
from antlr4 import *
from antlr4_impl.JavaLexer import JavaLexer
from antlr4_impl.JavaParser import JavaParser
from antlr4_impl.antlr4_utils import Antlr4Utils
import enchant


class CodeWordCountListener(JavaParserListener):
    def __init__(self):
        self.words = set()
        self.valid_words = set()
        self.invalid_words = set()
        self.spell_checker = enchant.Dict("en_US")

    def visitTerminal(self, node: TerminalNodeImpl):
        text = node.getText()
        words = self._split_identifier(text)

        for word in words:
            if not word.isdigit():
                if self.spell_checker.check(word):
                    self.valid_words.add(word)
                else:
                    self.invalid_words.add(word)
                self.words.add(word)

    def _split_identifier(self, text):
        """辅助方法：分割各种编程标识符"""
        words = []

        # 处理驼峰命名 (camelCase)
        if any(c.isupper() for c in text) and not text.isupper():
            current_word = [text[0].lower()]
            for c in text[1:]:
                if c.isupper():
                    words.append(''.join(current_word))
                    current_word = [c.lower()]
                else:
                    current_word.append(c)
            words.append(''.join(current_word))

        # 处理下划线命名 (snake_case)
        elif '_' in text:
            words = [w.lower() for w in text.split('_') if w]

        # 处理普通单词
        else:
            words = [text.lower()]

        return [w for w in words if w]  # 过滤空字符串


class JavaCodeWordCounter:
    def __init__(self, code):
        self.code = "".join([t for t in code.splitlines(keepends=True) if not t.startswith("import")])
        self.code = f"class _{{\n{code}\n}}"
        self.input_stream = InputStream(self.code)
        self.lexer = JavaLexer(self.input_stream)
        self.tokens = CommonTokenStream(self.lexer)
        self.parser = JavaParser(self.tokens)
        self.tree = self.parser.compilationUnit()
        self.tree = self.tree.children[0].children[0].children[2].children[1]
        self.listener = CodeWordCountListener()
        walker = ParseTreeWalker()
        walker.walk(self.listener, self.tree)

    def get_words(self):
        return self.listener.words

    def get_valid_words(self):
        return self.listener.valid_words

    def get_invalid_words(self):
        return self.listener.invalid_words


class CodeMetricsListener(JavaParserListener):
    operators = Antlr4JavaConst.operator_tokens  # 假设已定义

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
        self.spell_checker = enchant.Dict("en_US")
        self.valid_words = set()  # 完整单词计数
        self.invalid_words = set()  # 不完整单词计数
        self.code_seq = []  # 用于存储去除了注释的代码序列
        self.comment_seq = []  # 用于存储仅注释内容
        self.word_lengths = []

    def get_average_word_length(self):
        if not self.word_lengths:
            return 0.0
        return sum(self.word_lengths) / len(self.word_lengths)

    def visitTerminal(self, node: TerminalNodeImpl):
        line = node.symbol.line
        column = node.symbol.column
        text = node.getText()
        # 补充换行符：如果当前 token 的行号大于上一行，则补充 '\n'
        if self._last_line != -1 and line > self._last_line:
            lines_between = line - self._last_line
            for _ in range(lines_between - 1):
                self.code_seq.append('\n')
            self.code_seq.append('\n')

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

        # 新增：记录 code_seq 和 comment_seq
        if node.symbol.type in {JavaLexer.COMMENT, JavaLexer.LINE_COMMENT}:
            self.comment_seq.append(text)
        else:
            self.code_seq.append(text)

        # 原有逻辑继续保留
        if node.symbol.type in self.operators:
            self.operator_count += 1
        if node.symbol.type == JavaLexer.IDENTIFIER and self._is_real_operand(node):
            self.operand_count += 1
        if node.symbol.type not in {JavaLexer.COMMENT, JavaLexer.LINE_COMMENT,
                                    JavaLexer.WS} and node.symbol.line != self._last_line:
            self.line_count += 1
            self._last_line = node.symbol.line
        if node.symbol.type in {JavaLexer.STRING_LITERAL,
                                JavaLexer.DECIMAL_LITERAL, JavaLexer.HEX_LITERAL,
                                JavaLexer.OCT_LITERAL, JavaLexer.BINARY_LITERAL, JavaLexer.BOOL_LITERAL,
                                JavaLexer.FLOAT_LITERAL, JavaLexer.HEX_FLOAT_LITERAL,
                                JavaLexer.NULL_LITERAL}:
            self.operand_count += 1

        if node.symbol.type in {JavaLexer.IDENTIFIER, JavaLexer.STRING_LITERAL}:
            words = self._split_identifier(text)
            for word in words:
                if not word.isdigit():
                    if self.spell_checker.check(word):
                        self.valid_words.add(word)
                    else:
                        self.invalid_words.add(word)
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
        words = []

        # 处理驼峰命名 (camelCase)
        if any(c.isupper() for c in text) and not text.isupper():
            current_word = [text[0].lower()]
            for c in text[1:]:
                if c.isupper():
                    words.append(''.join(current_word))
                    current_word = [c.lower()]
                else:
                    current_word.append(c)
            words.append(''.join(current_word))

        # 处理下划线命名 (snake_case)
        elif '_' in text:
            words = [w.lower() for w in text.split('_') if w]

        # 处理普通单词
        else:
            words = [text.lower()]

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


import javalang
import networkx as nx
from collections import defaultdict


class JavaGraphBuilder:
    def __init__(self):
        self.ast = None
        self.cfg = nx.DiGraph()
        self.current_block = None
        self.block_stack = []
        self.edge_stack = []
        self.block_counter = 0
        self.label_map = defaultdict(list)

    def new_block(self, label=None):
        """创建一个新的基本块"""
        self.block_counter += 1
        block_id = f"{label} {self.block_counter}"
        self.cfg.add_node(block_id, statements=[], label=label)

        if label:
            self.label_map[label].append(block_id)

        return block_id

    def add_statement(self, stmt):
        """向当前块添加语句"""
        if self.current_block is None:
            self.current_block = self.new_block()
        self.cfg.nodes[self.current_block]['statements'].append(str(stmt))

    def connect_blocks(self, source, target):
        """连接两个基本块"""
        if source and target:
            self.cfg.add_edge(source, target)

    def build_cfg(self, java_code):
        """从Java代码构建CFG"""
        self.ast = javalang.parse.parse(java_code)

        for path, node in self.ast:
            if isinstance(node, javalang.tree.MethodDeclaration):
                self._process_method(node)

    def _process_method(self, method_node):
        """处理方法声明"""
        method_name = method_node.name
        # print(f"Building CFG for method: {method_name}")

        # 创建入口块
        entry_block = self.new_block(label="entry")
        self.current_block = entry_block

        # 处理方法体 - 处理可能为列表或BlockStatement的情况
        if hasattr(method_node, 'body'):
            self._process_block(method_node.body)
        elif hasattr(method_node, 'statements'):
            self._process_block(method_node.statements)

        # 创建出口块
        exit_block = self.new_block(label="exit")
        self.connect_blocks(self.current_block, exit_block)

    def _process_block(self, block_node):
        """处理代码块"""
        statements = []

        # 处理不同类型的block_node
        if block_node is None:
            return

        if isinstance(block_node, list):
            # 如果传入的是语句列表
            statements = block_node
        elif hasattr(block_node, 'statements'):
            # 如果是BlockStatement对象
            statements = block_node.statements
        else:
            # 其他情况（可能是单个语句）
            self._process_statement(block_node)
            return

        # 处理语句列表
        if statements:
            self.block_stack.append(self.current_block)

            for stmt in statements:
                self._process_statement(stmt)

            self.current_block = self.block_stack.pop()

    def _process_statement(self, stmt):
        """处理单个语句"""
        if isinstance(stmt, javalang.tree.IfStatement):
            self._process_if_statement(stmt)
        elif isinstance(stmt, javalang.tree.WhileStatement):
            self._process_while_statement(stmt)
        elif isinstance(stmt, javalang.tree.ForStatement):
            self._process_for_statement(stmt)
        elif isinstance(stmt, (javalang.tree.BlockStatement, list)):
            self._process_block(stmt)
        elif isinstance(stmt, javalang.tree.BreakStatement):
            self._process_break_statement(stmt)
        elif isinstance(stmt, javalang.tree.ContinueStatement):
            self._process_continue_statement(stmt)
        elif isinstance(stmt, javalang.tree.ReturnStatement):
            self._process_return_statement(stmt)
        else:
            self.add_statement(stmt)

    def _process_if_statement(self, if_node):
        """处理if语句"""
        # 添加条件表达式
        self.add_statement(if_node.condition)

        # 保存当前块作为if的起始块
        if_block = self.current_block

        # 创建then块
        then_block = self.new_block(label="if")
        self.connect_blocks(if_block, then_block)

        # 保存当前上下文
        self.edge_stack.append(('if', if_block, None))

        # 处理then块
        self.current_block = then_block
        self._process_block(if_node.then_statement)

        # 保存then块结束点
        then_end = self.current_block

        # 处理else块（如果有）
        else_end = None
        if if_node.else_statement:
            # 创建else块
            else_block = self.new_block(label="else")
            self.connect_blocks(if_block, else_block)

            # 更新上下文
            self.edge_stack[-1] = ('if', if_block, else_block)

            # 处理else块
            self.current_block = else_block
            self._process_block(if_node.else_statement)
            else_end = self.current_block

        # 创建合并块
        merge_block = self.new_block(label="merge")

        # 连接then和else到合并块
        self.connect_blocks(then_end, merge_block)
        if else_end:
            self.connect_blocks(else_end, merge_block)

        # 恢复上下文
        self.edge_stack.pop()
        self.current_block = merge_block

    def _process_while_statement(self, while_node):
        """处理while循环"""
        # 添加条件表达式
        self.add_statement(while_node.condition)

        # 保存当前块作为循环条件块
        condition_block = self.current_block

        # 创建循环体块
        body_block = self.new_block(label="while_body")
        self.connect_blocks(condition_block, body_block)

        # 保存当前上下文
        self.edge_stack.append(('while', condition_block))

        # 处理循环体
        self.current_block = body_block
        self._process_block(while_node.body)

        # 连接循环体回到条件
        self.connect_blocks(self.current_block, condition_block)

        # 创建退出块
        exit_block = self.new_block(label="while_exit")
        self.connect_blocks(condition_block, exit_block)

        # 恢复上下文
        self.edge_stack.pop()
        self.current_block = exit_block

    def _process_for_statement(self, for_node):
        """处理 javalang 的 ForStatement"""
        # 区分 basic for 和 enhanced for
        if hasattr(for_node, 'control'):
            # Basic for: for (init; condition; update)
            control = for_node.control

            # 初始化部分（init）
            if hasattr(control, 'init') and control.init:
                for init in control.init:
                    self.add_statement(init)

            # 创建条件块
            condition_block = self.new_block(label="for_condition")
            self.connect_blocks(self.current_block, condition_block)
            self.current_block = condition_block

            # 添加条件表达式
            if hasattr(control, 'condition') and control.condition:
                self.add_statement(control.condition)

            # 创建循环体块
            body_block = self.new_block(label="for_body")
            self.connect_blocks(condition_block, body_block)

            # 保存当前上下文
            self.edge_stack.append(('for', condition_block))
            self.current_block = body_block

            # 处理循环体
            self._process_statement(for_node.body)

            # 处理更新部分（update）
            if hasattr(control, 'update') and control.update:
                update_block = self.new_block(label="for_update")
                self.connect_blocks(self.current_block, update_block)
                self.current_block = update_block

                for update in control.update:
                    self.add_statement(update)

                # 回连到条件块
                self.connect_blocks(self.current_block, condition_block)
            else:
                # 直接回连到条件块
                self.connect_blocks(self.current_block, condition_block)

        elif hasattr(for_node, 'variable') and hasattr(for_node, 'iterable'):
            # Enhanced for: for (Type var : iterable)
            var_decl = for_node.variable
            iterable = for_node.iterable

            # 可选：记录变量声明和迭代对象
            # print(f"[EnhancedFor] Iterating over {iterable}, Variable: {var_decl}")

            # 创建条件块（模拟增强型 for 的控制流）
            condition_block = self.new_block(label="for_condition")
            self.connect_blocks(self.current_block, condition_block)
            self.current_block = condition_block
            self.add_statement(iterable)  # 添加 iterable 表达式作为条件的一部分

            # 创建循环体块
            body_block = self.new_block(label="for_body")
            self.connect_blocks(condition_block, body_block)
            self.current_block = body_block

            # 保存当前上下文
            self.edge_stack.append(('for', condition_block))

            # 处理循环体
            self._process_statement(for_node.body)

            # 模拟更新（无实际 update）
            self.connect_blocks(self.current_block, condition_block)

        else:
            # 未知 for 结构，直接处理循环体
            body_block = self.new_block(label="for_body")
            self.connect_blocks(self.current_block, body_block)
            self.current_block = body_block
            self.edge_stack.append(('for', self.current_block))
            self._process_statement(for_node.body)

        # 创建退出块
        exit_block = self.new_block(label="for_exit")
        self.connect_blocks(self.current_block, exit_block)

        # 恢复上下文
        self.edge_stack.pop()
        self.current_block = exit_block

    def _process_break_statement(self, break_node):
        """处理break语句"""
        # 查找最近的循环或switch
        for context in reversed(self.edge_stack):
            if context[0] in ('while', 'for', 'switch'):
                _, exit_point = context

                # 创建临时退出块
                temp_block = self.new_block(label="break_temp")
                self.connect_blocks(self.current_block, temp_block)

                # 标记需要连接到循环退出点
                self.label_map['break'].append((temp_block, exit_point))

                # 当前块结束
                self.current_block = None
                return

        # 如果没有找到循环上下文，直接结束当前块
        self.current_block = None

    def _process_continue_statement(self, continue_node):
        """处理continue语句"""
        # 类似于break，但连接回循环条件
        pass

    def _process_return_statement(self, return_node):
        """处理return语句"""
        if return_node.expression:
            self.add_statement(return_node.expression)

        # 当前块结束
        self.current_block = None


class KnownException(Exception):
    def __init__(self, msg):
        super(KnownException, self).__init__(msg)


class DecodeErrorException(Exception):
    def __init__(self, msg):
        super(DecodeErrorException, self).__init__(msg)


class SilentErrorListener(ErrorListener):
    def __init__(self):
        super().__init__()

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        # 静默处理错误，不输出任何内容
        pass


silent_listener = SilentErrorListener()


class JavaMatrixExtractor(AbstractMatrixExtractor):
    def __init__(self, code):
        code = "".join([t for t in code.splitlines(keepends=True) if not t.startswith("import")]).strip()
        t = code[:24]
        if " class " in t or " interface " in t or "import " in t:
            raise KnownException("Cannot extract features from class definition")
        del t
        code = f"class A{{\n{code}\n}}"
        super().__init__(code)
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
        # tokens = self.tokens.getTokens(0, -1)
        # for token in tokens:
        #     if token.channel == Lexer.HIDDEN:
        #         print(f"Hidden Token: {token.text}")
        self.parser = JavaParser(self.tokens)
        self.parser.removeErrorListeners()
        self.parser.addErrorListener(silent_listener)
        self.tree = self.parser.compilationUnit()
        self.tree = self.tree.children[0].children[0].children[2].children[1]
        self.listener = CodeMetricsListener()
        walker = ParseTreeWalker()
        walker.walk(self.listener, self.tree)
        self.graph_builder = JavaGraphBuilder()
        self.graph_builder.build_cfg(code)

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

    def get_number_of_comments(self):
        return self.listener.comment_count

    def get_number_of_comment_lines(self):
        return self.listener.comment_line_count

    def get_number_of_operands(self):
        return self.listener.operand_count

    def get_number_of_branches(self):
        return self.listener.branch_count

    def get_line_of_codes(self):
        return self.listener.line_count

    def get_comment_density(self):
        if self.listener.line_count == 0:
            return 0
        return len(self.comment_seq) / self.listener.line_count

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
        for method in methods:
            if method.startswith("get_") and callable(getattr(self, method)) and not method.startswith("get_token"):
                try:
                    value = getattr(self, method)()
                    feature_list.append(value)
                except Exception as e:
                    # 可选：记录错误或跳过无法调用的方法
                    logging.warning(f"Error calling {method}: {e}")
                    feature_list.append(None)
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
