#!/usr/bin/env python3
"""
Analyze ANY Python Repository - Generate summaries at function, module, and repo level

Usage:
    python analyze_any_repo.py <path_to_repo>

Example:
    python analyze_any_repo.py ../test_project
    python analyze_any_repo.py /path/to/django/project
    python analyze_any_repo.py .  (analyze current directory)
"""

import sys
import json
from pathlib import Path
from src.ast_parser import CodeAnalyzer
from src.function_summarizer import FunctionSummarizer
from src.module_summarizer import ModuleSummarizer
from src.repo_summarizer import RepoSummarizer
from src.module_grouper import ModuleGrouper
from src.diagram_generator import DiagramGenerator


def analyze_repository(repo_path: str):
    """Analyze any Python repository and output its structure"""

    repo_path_obj = Path(repo_path)

    if not repo_path_obj.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"ANALYZING REPOSITORY: {repo_path_obj.resolve()}")
    print("=" * 80)

    # Step 0: Run AST analysis
    print(f"\n Running AST analysis")
    analyzer = CodeAnalyzer(repo_path)
    results = analyzer.analyze_repository()
    summary = results.get('__analysis_summary__', {})

    # Step 1: Generate function summaries
    print(f"\n Step 1/4: Generating function summaries")
    function_summarizer = FunctionSummarizer()
    all_function_summaries = {}

    for module_path, module_data in results.items():
        if module_path == '__analysis_summary__' or 'error' in module_data:
            continue
        functions = module_data.get('functions', [])
        if functions:
            summaries = function_summarizer.summarize_module_functions(module_path, functions)
            if summaries:
                all_function_summaries[module_path] = summaries

    # Step 2: Generate module summaries
    print(f"\n Step 2/4: Generating module summaries")
    module_summarizer = ModuleSummarizer()
    module_summaries = module_summarizer.summarize_all_modules(results, all_function_summaries)

    # Step 3: Generate repository summary
    print(f"\n Step 3/4: Generating repository summary")
    repo_summarizer = RepoSummarizer()
    repo_summary = repo_summarizer.summarize_repository(
        repo_path_obj.name,
        module_summaries,
        summary
    )

    # Step 4: Generate architecture diagrams
    print(f"\n Step 4: Generating architecture diagrams")
    module_grouper = ModuleGrouper()
    logical_groups = module_grouper.group_modules(module_summaries)

    diagram_gen = DiagramGenerator(repo_path_obj.name)
    diagram_gen.generate_all_diagrams(logical_groups, results)

    # Save context file (used by Streamlit dashboard)
    context_file = f"context_{repo_path_obj.name}.json"
    context_data = {
        "repo_name": repo_path_obj.name,
        "ast_results": results,
        "function_summaries": all_function_summaries,
        "module_summaries": module_summaries,
        "repo_summary": repo_summary,
        "logical_groups": logical_groups
    }
    with open(context_file, 'w') as f:
        json.dump(context_data, f, indent=2)

    # Print final summary
    print(f"\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\n Modules: {summary.get('total_modules', 0)}")
    print(f" Functions: {summary.get('total_functions', 0)}")
    print(f" Classes: {summary.get('total_classes', 0)}")
    print(f"\n Output: {context_file} ({Path(context_file).stat().st_size:,} bytes)")
    print(f"\n Repository Summary:")
    print(f"   {repo_summary.get('human', 'N/A')}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    repo_path = sys.argv[1]
    analyze_repository(repo_path)
