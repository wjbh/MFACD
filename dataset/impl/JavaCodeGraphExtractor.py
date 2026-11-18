import os
import re
import sys
import traceback

import javalang

from itertools import count
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir)))
try:
    from dataset.impl.Exceptions import KnownException
except:
    from .Exceptions import KnownException

pattern = r'@[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?(?:$$(?:[^()]|\$$[^)]*$$)*$$)?\s*'

ast_node_type_id_map = {
    'CompilationUnit': 1,
    'Import': 2,
    'Documented': 3,
    'Declaration': 4,
    'TypeDeclaration': 5,
    'PackageDeclaration': 6,
    'ClassDeclaration': 7,
    'EnumDeclaration': 8,
    'InterfaceDeclaration': 9,
    'AnnotationDeclaration': 10,
    'Type': 11,
    'BasicType': 12,
    'ReferenceType': 13,
    'TypeArgument': 14,
    'TypeParameter': 15,
    'Annotation': 16,
    'ElementValuePair': 17,
    'ElementArrayValue': 18,
    'Member': 19,
    'MethodDeclaration': 20,
    'FieldDeclaration': 21,
    'ConstructorDeclaration': 22,
    'ConstantDeclaration': 23,
    'ArrayInitializer': 24,
    'VariableDeclaration': 25,
    'LocalVariableDeclaration': 26,
    'VariableDeclarator': 27,
    'FormalParameter': 28,
    'InferredFormalParameter': 29,
    'Statement': 30,
    'IfStatement': 31,
    'WhileStatement': 32,
    'DoStatement': 33,
    'ForStatement': 34,
    'AssertStatement': 35,
    'BreakStatement': 36,
    'ContinueStatement': 37,
    'ReturnStatement': 38,
    'ThrowStatement': 39,
    'SynchronizedStatement': 40,
    'TryStatement': 41,
    'SwitchStatement': 42,
    'BlockStatement': 43,
    'StatementExpression': 44,
    'TryResource': 45,
    'CatchClause': 46,
    'CatchClauseParameter': 47,
    'SwitchStatementCase': 48,
    'ForControl': 49,
    'EnhancedForControl': 50,
    'Expression': 51,
    'Assignment': 52,
    'TernaryExpression': 53,
    'BinaryOperation': 54,
    'Cast': 55,
    'MethodReference': 56,
    'LambdaExpression': 57,
    'Primary': 58,
    'Literal': 59,
    'This': 60,
    'MemberReference': 61,
    'Invocation': 62,
    'ExplicitConstructorInvocation': 63,
    'SuperConstructorInvocation': 64,
    'MethodInvocation': 65,
    'SuperMethodInvocation': 66,
    'SuperMemberReference': 67,
    'ArraySelector': 68,
    'ClassReference': 69,
    'VoidClassReference': 70,
    'Creator': 71,
    'ArrayCreator': 72,
    'ClassCreator': 73,
    'InnerClassCreator': 74,
    'EnumBody': 75,
    'EnumConstantDeclaration': 76,
    'AnnotationMethod': 77,
}


class JavaCodeGraphExtractor:
    def __init__(self, code):
        code = "".join([t for t in code.splitlines(keepends=True) if not t.startswith("import")]).strip()
        t = code[:24]
        if " class " in t or " interface " in t or "import " in t:
            raise KnownException("Cannot extract features from class definition")
        del t

        code = re.sub(pattern, '', code)
        code = f"class A{{\n{code}\n}}"
        try:
            self.ast = javalang.parse.parse(code)
        except :
            raise KnownException("Cannot parse code")
        for path, node in self.ast.filter(javalang.tree.MethodDeclaration):
            self.ast = node
            break

    def __traverse(self, node, parent_id=None, vertices=None, edges=None, index_provider=None):
        if vertices is None:
            vertices = dict()
        if edges is None:
            edges = set()
        if index_provider is None:
            index_provider = count(start=0)
        node_id = next(index_provider)

        typename = str(type(node))[8:-2]
        vertices[node_id] = ast_node_type_id_map[typename[typename.rfind('.') + 1:]]

        if parent_id is not None:
            edges.add((parent_id, node_id))

        for field_name in node.attrs:
            field_value = getattr(node, field_name)
            if isinstance(field_value, list):
                for child in field_value:
                    if isinstance(child, javalang.ast.Node):
                        self.__traverse(child, node_id, vertices=vertices, edges=edges, index_provider=index_provider)
            elif isinstance(field_value, javalang.ast.Node):
                self.__traverse(field_value, node_id, vertices=vertices, edges=edges, index_provider=index_provider)
        return vertices, edges

    def get_ast_feature(self):
        return self.__traverse(self.ast)


if __name__ == '__main__':
    __code = """
private static <T> @NonNull List<T> topologicalSort(final @NonNull Graph<T> graph, final @NonNull SortType<T> type) {
    checkArgument(graph.isDirected(), "the graph must be directed");
    checkArgument(!graph.allowsSelfLoops(), "the graph cannot allow self loops");
    final Map<T, Integer> requiredCounts = new HashMap<>();
    for(final T node : graph.nodes()) {
      for(final T successor : graph.successors(node)) {
        requiredCounts.merge(successor, 1, (a, b) -> a + b);
      }
    }
    final Queue<T> processing = type.createQueue();
    final List<T> results = new ArrayList<>();
    for(final T node : graph.nodes()) {
      if(!requiredCounts.containsKey(node)) {
        processing.add(node);
      }
    }
    while(!processing.isEmpty()) {
      final T now = processing.poll();
      for(final T successor : graph.successors(now)) {
        final int newCount = requiredCounts.get(successor) - 1;
        if(newCount == 0) {
          processing.add(successor);
          requiredCounts.remove(successor);
        } else {
          requiredCounts.put(successor, newCount);
        }
      }
      results.add(now);
    }
    if(!requiredCounts.isEmpty()) {
      final StronglyConnectedComponentAnalyzer<T> analyzer = new StronglyConnectedComponentAnalyzer<>(graph);
      analyzer.analyze();
      throw new CyclePresentException("Graph (" + graph + ") has cycle(s): " + analyzer.renderCycles(), analyzer.components());
    }
    return results;
  }

    """
    __ex = JavaCodeGraphExtractor(__code)
    print(__ex.get_ast_feature())
