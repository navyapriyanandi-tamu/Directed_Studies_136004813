#!/usr/bin/env python3
"""
CodeChat - Unified Entry Point
Analyzes a repository and launches interactive Streamlit dashboard in one command

Usage:
    python3 codechat.py <repo_path>              # Full analysis + Streamlit
    python3 codechat.py <repo_path> --skip-analysis  # Streamlit only (use existing context)
"""

import sys
import subprocess
from pathlib import Path
import json


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 codechat.py <repo_path> [--skip-analysis]")
        print("\nExamples:")
        print("  python3 codechat.py ../test_project           # Full analysis + Streamlit")
        print("  python3 codechat.py test_project --skip-analysis  # Streamlit only")
        sys.exit(1)

    repo_path = sys.argv[1]
    skip_analysis = "--skip-analysis" in sys.argv

    # Determine repo name
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists():
        print(f" Error: Repository path not found: {repo_path}")
        sys.exit(1)

    repo_name = repo_path_obj.name

    # Check if context exists
    context_file = Path(f"context_{repo_name}.json")

    if skip_analysis:
        print(f" Skipping analysis, using existing context")
        if not context_file.exists():
            print(f" Error: No context file found: {context_file}")
            print(f" Run without --skip-analysis to generate context first")
            sys.exit(1)
    else:
        # Step 1: Run analysis
        print("=" * 80)
        print(f" STEP 1: Analyzing Repository: {repo_path}")
        print("=" * 80)

        result = subprocess.run(
            ["python3", "analyze_any_repo.py", repo_path],
            capture_output=False
        )

        if result.returncode != 0:
            print(f"\n Analysis failed with exit code {result.returncode}")
            sys.exit(1)

        if not context_file.exists():
            print(f"\n Error: Context file was not created: {context_file}")
            sys.exit(1)

        print("\n" + "=" * 80)
        print(" Analysis Complete!")
        print("=" * 80)

    # Step 2: Launch Streamlit Dashboard
    print(f"\n STEP 2: Launching Streamlit Dashboard")
    print("=" * 80)
    print(f"\n The dashboard will open in your browser automatically.")
    print(f"   Repository: {repo_name}")
    print(f"   Context: {context_file}")
    print("\n" + "=" * 80)

    # Launch Streamlit with repo name as argument
    result = subprocess.run(
        ["streamlit", "run", "streamlit_app.py", "--", repo_name],
        capture_output=False
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
