from antlr4_impl.JavaLexer import JavaLexer


class Antlr4JavaConst:
    operator_tokens = {
        # basic operator
        JavaLexer.ASSIGN,  # =
        JavaLexer.GT,  # >
        JavaLexer.LT,  # <
        JavaLexer.BANG,  # !
        JavaLexer.TILDE,  # ~
        JavaLexer.QUESTION,  # ?
        JavaLexer.COLON,  # :
        JavaLexer.EQUAL,  # ==
        JavaLexer.LE,  # <=
        JavaLexer.GE,  # >=
        JavaLexer.NOTEQUAL,  # !=
        JavaLexer.AND,  # &&
        JavaLexer.OR,  # ||
        JavaLexer.INC,  # ++
        JavaLexer.DEC,  # --
        JavaLexer.ADD,  # +
        JavaLexer.SUB,  # -
        JavaLexer.MUL,  # *
        JavaLexer.DIV,  # /
        JavaLexer.BITAND,  # &
        JavaLexer.BITOR,  # |
        JavaLexer.CARET,  # ^
        JavaLexer.MOD,  # %

        # other
        JavaLexer.ADD_ASSIGN,  # +=
        JavaLexer.SUB_ASSIGN,  # -=
        JavaLexer.MUL_ASSIGN,  # *=
        JavaLexer.DIV_ASSIGN,  # /=
        JavaLexer.AND_ASSIGN,  # &=
        JavaLexer.OR_ASSIGN,  # |=
        JavaLexer.XOR_ASSIGN,  # ^=
        JavaLexer.MOD_ASSIGN,  # %=
        JavaLexer.LSHIFT_ASSIGN,  # <<=
        JavaLexer.RSHIFT_ASSIGN,  # >>=
        JavaLexer.URSHIFT_ASSIGN,  # >>>=
    }
    operand_tokens = {
        JavaLexer.STRING_LITERAL,
        JavaLexer.DECIMAL_LITERAL,
        JavaLexer.HEX_LITERAL,
        JavaLexer.OCT_LITERAL,
        JavaLexer.BINARY_LITERAL,
        JavaLexer.BOOL_LITERAL,
        JavaLexer.FLOAT_LITERAL,
        JavaLexer.HEX_FLOAT_LITERAL,
        JavaLexer.NULL_LITERAL
    }
    comment_tokens = {
        JavaLexer.COMMENT,
        JavaLexer.LINE_COMMENT
    }
