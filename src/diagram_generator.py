"""
Architecture Diagram Generator
Creates visual diagrams of repository architecture
"""

from typing import Dict, List, Set, Tuple
from pathlib import Path
import os

class DiagramGenerator:
    """
    Generates architecture diagrams from code analysis.
    Output: ASCII art for terminals and Mermaid.js syntax for documentation.
    """

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        self.output_dir = Path("diagrams")
        self.output_dir.mkdir(exist_ok=True)

    def generate_all_diagrams(
        self,
        logical_groups: Dict[str, List[str]],
        ast_results: Dict
    ) -> Dict[str, str]:
        """
        Generate diagram types: Dependencies (ASCII) and Mermaid (visual)
        """
        print(f"\n Generating Architecture Diagrams for {self.repo_name}")

        diagrams = {}

        # 0. FALLBACK: If logical_groups is missing or empty, group by folders

        if not logical_groups:
            print("    'logical_groups' key missing or empty. Falling back to folder-based grouping.")
            logical_groups = {}
            for file_path in ast_results.keys():
                if file_path == '__analysis_summary__': continue

                # Use the parent folder name as the group

                path_obj = Path(file_path)
                folder = path_obj.parent.name

                # Handle root files (parent is empty or '.')
                if not folder or folder == '.':
                    folder = "Root / Top Level"

                if folder not in logical_groups:
                    logical_groups[folder] = []
                logical_groups[folder].append(file_path)

        # 1. Extract Dependencies

        dependencies = self._extract_dependencies(ast_results)

        # 2. Generate ASCII Dependency Tree
        print("     Generating dependency graph (ASCII)")
        dep_ascii = self._generate_ascii_dependencies(ast_results, dependencies)
        self._save_diagram(f"dependencies_{self.repo_name}.txt", dep_ascii)
        diagrams['ascii_deps'] = dep_ascii

        # 3. Generate Mermaid.js (For GitHub/Markdown)
        print("   Generating Mermaid.js visual diagram")
        mermaid_code = self._generate_mermaid(logical_groups, dependencies)
        self._save_diagram(f"mermaid_{self.repo_name}.mmd", mermaid_code)
        diagrams['mermaid'] = mermaid_code

        print(f"    Diagrams saved to: {self.output_dir.absolute()}/")
        print(f"    dependencies_{self.repo_name}.txt")
        print(f"    mermaid_{self.repo_name}.mmd")

        return diagrams

    def _extract_dependencies(self, ast_results: Dict) -> Dict[str, Set[str]]:
        """
        Robustly links imports to actual files in the analysis.
        Handles both simple imports (import model) and dotted imports (from task_manager.task_handler import X)
        """
        dependencies = {}

        # Create comprehensive lookup maps

        file_lookup = {}
        for file_key in ast_results.keys():
            if file_key == '__analysis_summary__': continue

            # Map the stem (filename without extension) to the full key
            stem = Path(file_key).stem
            file_lookup[stem] = file_key

            # Also map the full path without extension
            path_without_ext = str(Path(file_key).with_suffix(''))
            file_lookup[path_without_ext] = file_key

            # Map the dotted version (task_manager/task_handler -> task_manager.task_handler)
            dotted_path = path_without_ext.replace('/', '.')
            file_lookup[dotted_path] = file_key

        for module_path, data in ast_results.items():
            if module_path == '__analysis_summary__': continue

            deps = set()
            imports = data.get('imports', [])

            for imp in imports:
                imported_name = imp.get('module') or imp.get('name')
                if not imported_name: continue

                # Try to resolve the import in order of specificity:
                # 1. Exact match (task_manager.task_handler)
                # 2. Base module match (task_manager)

                if imported_name in file_lookup:
                    # Direct match
                    deps.add(file_lookup[imported_name])
                else:
                    # Try base module (first part before dot)
                    base_module = imported_name.split('.')[0]
                    if base_module in file_lookup:
                        deps.add(file_lookup[base_module])

            if deps:
                dependencies[module_path] = deps

        return dependencies

    def _generate_ascii_dependencies(
        self,
        ast_results: Dict,
        dependencies: Dict[str, Set[str]]
    ) -> str:
        """
        Draws a text-based tree of imports
        """
        lines = []
        lines.append(" MODULE DEPENDENCY GRAPH")
        lines.append("=" * 60)

        files = sorted([k for k in ast_results.keys() if k != '__analysis_summary__'])

        for f in files:
            fname = Path(f).name
            if f not in dependencies:
                continue

            lines.append(f" {fname}")
            deps = sorted(dependencies[f])

            for i, dep in enumerate(deps):
                dep_name = Path(dep).name
                is_last = (i == len(deps) - 1)
                prefix = "  └─>" if is_last else "  ├─>"
                lines.append(f"{prefix} {dep_name}")
            lines.append("")

        return "\n".join(lines)

    def _generate_mermaid(
        self,
        logical_groups: Dict[str, List[str]],
        dependencies: Dict[str, Set[str]]
    ) -> str:
        """
        Generates Mermaid.js code for professional rendering with color-coded groups

        """
        # Count total potential edges
        total_edges = sum(len(targets) for targets in dependencies.values())

        
        print(f"   Repository has {total_edges} file-level dependencies")
        print(f"   Using simplified view: showing connections between logical groups")

        return self._generate_simplified_mermaid(logical_groups, dependencies)

    def _generate_simplified_mermaid(
        self,
        logical_groups: Dict[str, List[str]],
        dependencies: Dict[str, Set[str]]
    ) -> str:
        
        mmd = ["graph TD"]
        # Sanitize repo name - 'click' and 'end' are reserved keywords in Mermaid
        safe_repo_name = self.repo_name.replace("click", "click_repo").replace("end", "end_repo")
        mmd.append(f"    subgraph {safe_repo_name}")

        # Color scheme
        def get_group_color(group_name: str) -> str:
            name_lower = group_name.lower()
            if any(k in name_lower for k in ['entry', 'main', 'application']): return '#4A90E2'
            elif any(k in name_lower for k in ['core', 'api', 'logic', 'business', 'handler']): return '#50C878'
            elif any(k in name_lower for k in ['model', 'data', 'storage', 'database']): return '#9B59B6'
            elif any(k in name_lower for k in ['network', 'connection', 'adapter', 'transport']): return '#E67E22'
            elif any(k in name_lower for k in ['auth', 'security', 'cert']): return '#E74C3C'
            elif any(k in name_lower for k in ['util', 'helper', 'tool']): return '#F39C12'
            elif any(k in name_lower for k in ['error', 'exception', 'diagnostic']): return '#C0392B'
            elif any(k in name_lower for k in ['doc', 'setup', 'config', 'distribution', 'package']): return '#7F8C8D'
            elif any(k in name_lower for k in ['init', 'compat', 'library']): return '#5DADE2'
            elif any(k in name_lower for k in ['extension', 'hook', 'plugin']): return '#16A085'
            else: return '#95A5A6'

        # Helper to create SAFE Node IDs (short, alphanumeric only)
        def clean_id(text: str) -> str:
            # Use hash to create short, unique IDs (avoid long node names that Mermaid rejects)
            import hashlib
            # Create a short hash of the group name
            hash_val = hashlib.md5(text.encode()).hexdigest()[:8]
            # Use first word + hash for readability
            first_word = text.split()[0] if text else "group"
            # Clean the first word
            first_word = first_word.replace("&", "and").replace("-", "").replace(".", "")
            return f"g_{first_word}_{hash_val}"

        # Helper to create SAFE Display Labels (plain text only)
        def clean_label(text: str) -> str:
            # Remove/replace characters that Mermaid interprets as syntax
            return text.replace("&", "and").replace('"', "'").replace(":", " -")

        # Create a mapping from file to group
        file_to_group = {}
        for group_name, files in logical_groups.items():
            for f in files:
                file_to_group[f] = group_name

        # Create group nodes
        for group_name, files in logical_groups.items():
            node_id = clean_id(group_name)
            color = get_group_color(group_name)
            safe_title = clean_label(group_name)

            # Mermaid has a strict label length limit (~40 chars for labels)
            # Show group name with truncated file list
            files_sorted = sorted([Path(f).name for f in files])

            # Show all files if <=5, otherwise show first 5 + count
            MAX_FILES = 5
            base_label = f"{safe_title} - "

            if len(files_sorted) <= MAX_FILES:
                # 5 files or fewer - show ALL of them (no char limit)
                file_list = files_sorted
                node_label = base_label + ", ".join(file_list)
            else:
                # More than 5 files - show first 5 + count
                file_list = files_sorted[:MAX_FILES]
                remaining = len(files_sorted) - MAX_FILES
                file_list.append(f"+{remaining} more")
                node_label = base_label + ", ".join(file_list)

            mmd.append(f"        {node_id}[\"{node_label}\"]")
            mmd.append(f"        style {node_id} fill:{color},stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF")

        mmd.append("")

        # Calculate group-to-group dependencies
        group_connections = {}

        for source_file, target_files in dependencies.items():
            if source_file not in file_to_group: continue
            source_group = file_to_group[source_file]

            if source_group not in group_connections:
                group_connections[source_group] = set()

            for target_file in target_files:
                if target_file not in file_to_group: continue
                target_group = file_to_group[target_file]

                if source_group != target_group:
                    group_connections[source_group].add(target_group)

        # Add edges
        for source_group, target_groups in group_connections.items():
            src_id = clean_id(source_group)
            for target_group in target_groups:
                tgt_id = clean_id(target_group)
                mmd.append(f"    {src_id} --> {tgt_id}")

        mmd.append("")
        mmd.append("    linkStyle default stroke:#FFFFFF,stroke-width:2px")
        mmd.append("")
        mmd.append("    end")

        return "\n".join(mmd)

    def _save_diagram(self, filename: str, content: str):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
