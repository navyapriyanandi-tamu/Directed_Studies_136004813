"""
AST Parser for Python Repositories
Analyzes code structure for summary generation
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Optional
import config


class CodeAnalyzer:
    """
    code analyzer using AST
    Extracts functions, classes, imports, and dependencies from repositories
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.modules: Dict[str, Dict] = {}
        self.all_defined_functions: Set[str] = set()
        self.all_defined_classes: Set[str] = set()

    def analyze_repository(self) -> Dict[str, Dict]:
        """Analyze entire repository"""
        print(f" Analyzing repository: {self.repo_path}")

        # Find all Python files
        python_files = self._find_python_files()
        print(f"   Found {len(python_files)} Python files")

        # Analyze each file
        for py_file in python_files:
            relative_path = str(py_file.relative_to(self.repo_path))
            print(f"   Analyzing: {relative_path}")

            try:
                analysis = self.analyze_file(py_file)
                if analysis:
                    self.modules[relative_path] = analysis

                    # Track globally defined items
                    for func in analysis.get('functions', []):
                        self.all_defined_functions.add(func['name'])
                    for cls in analysis.get('classes', []):
                        self.all_defined_classes.add(cls['name'])

            except Exception as e:
                print(f"    Error analyzing {relative_path}: {e}")
                self.modules[relative_path] = {'error': str(e)}

        # Add summary
        self.modules['__analysis_summary__'] = {
            'total_modules': len(self.modules) - 1,
            'total_functions': len(self.all_defined_functions),
            'total_classes': len(self.all_defined_classes)
        }

        print(f"\n Analysis complete!")
        print(f"   Modules: {len(self.modules) - 1}")
        print(f"   Functions: {len(self.all_defined_functions)}")
        print(f"   Classes: {len(self.all_defined_classes)}")

        return self.modules

    def analyze_file(self, filepath: Path) -> Optional[Dict]:
        """Analyze a single Python file using AST"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code, filename=str(filepath))

            # Extract all components
            module_docstring = ast.get_docstring(tree)
            functions = self._extract_functions(tree, code)
            classes = self._extract_classes(tree)
            imports = self._extract_imports(tree)
            variables = self._extract_module_variables(tree)
            is_entry_point = self._is_entry_point(tree)
            main_block_calls = self._extract_main_block_calls(tree) if is_entry_point else []
            module_level_calls = self._extract_module_level_calls(tree)

            return {
                'filepath': str(filepath),
                'module_docstring': module_docstring,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'variables': variables,
                'is_entry_point': is_entry_point,
                'main_block_calls': main_block_calls,
                'module_level_calls': module_level_calls,
                'line_count': len(code.split('\n'))
            }

        except (SyntaxError, Exception):
            return None

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in repository"""
        python_files = []
        skip_patterns = set(config.SKIP_PATTERNS)

        for py_file in self.repo_path.rglob("*.py"):
            if any(skip in py_file.parts for skip in skip_patterns):
                continue
            python_files.append(py_file)

        return sorted(python_files)

    def _extract_functions(self, tree, source_code: str) -> List[Dict]:
        """Extract all function definitions with type hints, decorators, and full code"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract arguments with type hints
                args_with_types = []
                for arg in node.args.args:
                    arg_info = {'name': arg.arg}
                    if arg.annotation:
                        arg_info['type'] = self._get_annotation_string(arg.annotation)
                    args_with_types.append(arg_info)

                # Extract return type
                return_type = None
                if node.returns:
                    return_type = self._get_annotation_string(node.returns)

                # Extract decorators with parameters
                decorators = []
                for dec in node.decorator_list:
                    decorators.append(self._extract_decorator_info(dec))

                # Extract function calls and references
                func_calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        call_name = self._get_call_name(child)
                        if call_name:
                            func_calls.append(call_name)

                        for arg in child.args:
                            if isinstance(arg, ast.Name):
                                func_calls.append(arg.id)
                            elif isinstance(arg, ast.Attribute):
                                func_name = self._get_name(arg)
                                if func_name:
                                    func_calls.append(func_name)

                        for keyword in child.keywords:
                            if isinstance(keyword.value, ast.Name):
                                func_calls.append(keyword.value.id)
                            elif isinstance(keyword.value, ast.Attribute):
                                func_name = self._get_name(keyword.value)
                                if func_name:
                                    func_calls.append(func_name)

                    elif isinstance(child, ast.Assign):
                        if isinstance(child.value, ast.Name):
                            func_calls.append(child.value.id)
                        elif isinstance(child.value, ast.Attribute):
                            func_name = self._get_name(child.value)
                            if func_name:
                                func_calls.append(func_name)

                # Extract full function code
                func_code = ast.get_source_segment(source_code, node)
                if not func_code:
                    func_code = ""

                func_line_count = len(func_code.split('\n')) if func_code else 0
                needs_llm_summary = func_line_count >= 3

                functions.append({
                    'name': node.name,
                    'args': args_with_types,
                    'return_type': return_type,
                    'lineno': node.lineno,
                    'decorators': decorators,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'calls': list(set(func_calls)),
                    'docstring': ast.get_docstring(node),
                    'code': func_code,
                    'line_count': func_line_count,
                    'needs_llm_summary': needs_llm_summary
                })

        return functions

    def _extract_classes(self, tree) -> List[Dict]:
        """Extract all class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(child.name)

                classes.append({
                    'name': node.name,
                    'methods': methods,
                    'base_classes': [self._get_name(base) for base in node.bases],
                    'lineno': node.lineno,
                    'docstring': ast.get_docstring(node)
                })

        return classes

    def _extract_imports(self, tree) -> List[Dict]:
        """Extract all imports"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'lineno': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'lineno': node.lineno,
                        'type': 'from_import'
                    })

        return imports

    def _extract_module_variables(self, tree) -> List[Dict]:
        """Extract module-level variables with their assignments"""
        variables = []

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_to = None
                        if isinstance(node.value, ast.Call):
                            assigned_to = self._get_call_name(node.value)
                        elif isinstance(node.value, ast.Name):
                            assigned_to = node.value.id
                        elif isinstance(node.value, ast.Constant):
                            assigned_to = type(node.value.value).__name__

                        variables.append({
                            'name': target.id,
                            'assigned_to': assigned_to,
                            'lineno': node.lineno
                        })

        return variables

    def _is_entry_point(self, tree) -> bool:
        """Check if module is an entry point"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                        return True
        return False

    def _extract_main_block_calls(self, tree) -> List[str]:
        """Extract function calls from __main__ block"""
        main_block_calls = []

        for node in tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                call_name = self._get_call_name(child)
                                if call_name:
                                    main_block_calls.append(call_name)

        return list(set(main_block_calls))

    def _extract_module_level_calls(self, tree) -> Dict:
        """Extract function/method calls and class instantiations at module level"""
        module_calls = {
            'function_calls': [],
            'class_instantiations': []
        }

        skip_nodes = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                skip_nodes.add(id(node))

        for node in tree.body:
            if id(node) in skip_nodes:
                continue

            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_name = self._get_call_name(child)
                    if call_name:
                        if call_name and call_name[0].isupper():
                            base_name = call_name.split('.')[0]
                            module_calls['class_instantiations'].append(base_name)
                        module_calls['function_calls'].append(call_name)

                    for arg in child.args:
                        if isinstance(arg, ast.Name):
                            module_calls['function_calls'].append(arg.id)
                        elif isinstance(arg, ast.Attribute):
                            func_name = self._get_name(arg)
                            if func_name:
                                module_calls['function_calls'].append(func_name)

                    for keyword in child.keywords:
                        if isinstance(keyword.value, ast.Name):
                            module_calls['function_calls'].append(keyword.value.id)
                        elif isinstance(keyword.value, ast.Attribute):
                            func_name = self._get_name(keyword.value)
                            if func_name:
                                module_calls['function_calls'].append(func_name)

                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(child.value, ast.Name):
                            module_calls['function_calls'].append(child.value.id)
                        elif isinstance(child.value, ast.Attribute):
                            func_name = self._get_name(child.value)
                            if func_name:
                                module_calls['function_calls'].append(func_name)

        module_calls['function_calls'] = list(set(module_calls['function_calls']))
        module_calls['class_instantiations'] = list(set(module_calls['class_instantiations']))

        return module_calls

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Get function call name"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_name(node.func)
        return None

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def _get_annotation_string(self, node) -> str:
        """Convert type annotation to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_string(node.value)
            slice_val = self._get_annotation_string(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_annotation_string(e) for e in node.elts]
            return ", ".join(elements)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_string(node.value)}.{node.attr}"
        return "Any"

    def _extract_decorator_info(self, node) -> Dict:
        """Extract decorator information including parameters"""
        if isinstance(node, ast.Name):
            return {'name': node.id, 'args': [], 'kwargs': {}}

        elif isinstance(node, ast.Call):
            decorator_name = self._get_name(node.func)

            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                elif isinstance(arg, ast.List):
                    list_items = [item.value for item in arg.elts if isinstance(item, ast.Constant)]
                    args.append(list_items)

            kwargs = {}
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    kwargs[keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.List):
                    list_items = [item.value for item in keyword.value.elts if isinstance(item, ast.Constant)]
                    kwargs[keyword.arg] = list_items

            return {'name': decorator_name, 'args': args, 'kwargs': kwargs}

        elif isinstance(node, ast.Attribute):
            return {'name': self._get_name(node), 'args': [], 'kwargs': {}}

        return {'name': 'unknown', 'args': [], 'kwargs': {}}

# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ast_parser <path_to_repo>")
        sys.exit(1)

    repo_path = sys.argv[1]
    if not Path(repo_path).exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    analyzer = CodeAnalyzer(repo_path)
    results = analyzer.analyze_repository()

    summary = results.get('__analysis_summary__', {})
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total modules: {summary.get('total_modules', 0)}")
    print(f"Total functions: {summary.get('total_functions', 0)}")
    print(f"Total classes: {summary.get('total_classes', 0)}")
