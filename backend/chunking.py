from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import esprima


MAX_CHUNK_LINES = 120


@dataclass
class CodeChunk:
    file_path: str
    language: str
    type: str
    name: str
    code: str
    line_start: int
    line_end: int
    parent: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_large_chunk(chunk: CodeChunk) -> List[CodeChunk]:
    lines = chunk.code.splitlines()
    if len(lines) <= MAX_CHUNK_LINES:
        return [chunk]

    split_chunks: List[CodeChunk] = []
    current_start = chunk.line_start

    for index in range(0, len(lines), MAX_CHUNK_LINES):
        part_lines = lines[index : index + MAX_CHUNK_LINES]
        part_start = current_start + index
        part_end = part_start + len(part_lines) - 1
        split_chunks.append(
            CodeChunk(
                file_path=chunk.file_path,
                language=chunk.language,
                type=f"{chunk.type}_part",
                name=f"{chunk.name}#part{(index // MAX_CHUNK_LINES) + 1}",
                code="\n".join(part_lines),
                line_start=part_start,
                line_end=part_end,
                parent=chunk.parent,
            )
        )

    return split_chunks


def _deduplicate(chunks: List[CodeChunk]) -> List[CodeChunk]:
    seen = set()
    unique: List[CodeChunk] = []
    for chunk in chunks:
        key = (chunk.file_path, chunk.type, chunk.name, chunk.line_start, chunk.line_end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _extract_python_chunks(code: str, file_path: str) -> List[CodeChunk]:
    tree = ast.parse(code)
    lines = code.splitlines()
    chunks: List[CodeChunk] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: List[str] = []

        def _build_chunk(self, node: ast.AST, item_type: str, name: str, parent: str | None = None) -> None:
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            snippet = "\n".join(lines[start - 1 : end])
            chunks.extend(
                split_large_chunk(
                    CodeChunk(
                        file_path=file_path,
                        language="python",
                        type=item_type,
                        name=name,
                        code=snippet,
                        line_start=start,
                        line_end=end,
                        parent=parent,
                    )
                )
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            parent = self.class_stack[-1] if self.class_stack else None
            item_type = "method" if parent else "function"
            display_name = f"{parent}.{node.name}" if parent else node.name
            self._build_chunk(node, item_type, display_name, parent)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            parent = self.class_stack[-1] if self.class_stack else None
            item_type = "method" if parent else "function"
            display_name = f"{parent}.{node.name}" if parent else node.name
            self._build_chunk(node, item_type, display_name, parent)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            self._build_chunk(node, "class", node.name)
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

    Visitor().visit(tree)
    return _deduplicate(chunks)


def _js_name_from_declaration(node: Any) -> str | None:
    if getattr(node, "id", None) and getattr(node.id, "name", None):
        return node.id.name
    return None


def _extract_javascript_chunks(code: str, file_path: str) -> List[CodeChunk]:
    program = esprima.parseModule(code, loc=True, tolerant=True)
    lines = code.splitlines()
    chunks: List[CodeChunk] = []

    def build_chunk(node: Any, item_type: str, name: str, parent: str | None = None) -> None:
        start = node.loc.start.line
        end = node.loc.end.line
        snippet = "\n".join(lines[start - 1 : end])
        chunks.extend(
            split_large_chunk(
                CodeChunk(
                    file_path=file_path,
                    language="javascript",
                    type=item_type,
                    name=name,
                    code=snippet,
                    line_start=start,
                    line_end=end,
                    parent=parent,
                )
            )
        )

    for node in program.body:
        node_type = getattr(node, "type", "")

        if node_type == "FunctionDeclaration":
            name = _js_name_from_declaration(node) or "anonymous_function"
            build_chunk(node, "function", name)
            continue

        if node_type == "ClassDeclaration":
            class_name = _js_name_from_declaration(node) or "AnonymousClass"
            build_chunk(node, "class", class_name)
            for element in getattr(node.body, "body", []):
                if getattr(element, "type", "") == "MethodDefinition":
                    method_name = getattr(getattr(element, "key", None), "name", "anonymous_method")
                    build_chunk(element.value, "method", f"{class_name}.{method_name}", class_name)
            continue

        if node_type == "VariableDeclaration":
            for declaration in getattr(node, "declarations", []):
                init = getattr(declaration, "init", None)
                identifier = getattr(getattr(declaration, "id", None), "name", None)
                init_type = getattr(init, "type", "")

                if identifier and init_type in {"ArrowFunctionExpression", "FunctionExpression"}:
                    build_chunk(init, "function", identifier)

    return _deduplicate(chunks)


def extract_code_chunks(code: str, file_path: str, language: str) -> List[Dict[str, Any]]:
    if language == "python":
        chunks = _extract_python_chunks(code, file_path)
    elif language == "javascript":
        chunks = _extract_javascript_chunks(code, file_path)
    else:
        chunks = []

    return [chunk.to_dict() for chunk in chunks]
