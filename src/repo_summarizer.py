"""
Repository Summarizer - Generate repository-level summary from module summaries
Aggregates all module summaries to create a cohesive repository-level summary
"""

import json
from typing import Dict, Optional
import config


class RepoSummarizer:
    """
    Generate summary for entire repository using module summaries
    """

    
    MAX_MODULES_PER_CALL = 40  # Max modules to include in one API call

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with TAMU AI Chat API key"""
        self.client = config.get_client()
        self.model = config.LLM_MODEL

        # Track API calls
        self.api_calls = 0

    def summarize_repository(
        self,
        repo_name: str,
        module_summaries: Dict[str, Dict[str, str]],
        ast_summary: Dict
    ) -> Dict[str, str]:
        """
        Summarize entire repository
        """

        if not module_summaries:
            return {
                "human": f"Repository: {repo_name}",
                "technical": "No module summaries available for repository-level summarization"
            }

        print(f"\n  Generating repository summary for '{repo_name}'")
        print(f"   Aggregating {len(module_summaries)} module summaries")

        # Check if we need to chunk (very large repo with many modules)
        if len(module_summaries) > self.MAX_MODULES_PER_CALL:
            return self._summarize_large_repo(repo_name, module_summaries, ast_summary)

        # Build repository context
        repo_context = self._build_repo_context(repo_name, ast_summary)

        # Format module summaries
        modules_text = self._format_module_summaries(module_summaries)

        # Generate prompt
        prompt = f"""You are a senior Python engineer writing the README-level overview for a codebase you've just finished reviewing. 
        Another developer is about to start working in this repository for the first time, and your summary is the only context they will have before they open any file.
        
        Analyze this Python repository and provide a TWO-LEVEL summary.

{repo_context}

**Module Summaries:**
{modules_text}

---

Treat the module summaries above as ground truth. Do not invent components, classes, or design patterns that aren't reflected there. If the module summaries don't mention a feature, your summary shouldn't claim it either.

Based on the module summaries above, provide TWO summaries for the ENTIRE REPOSITORY:

**Human Summary:**
[Explain what this repository does in plain language. What is its purpose? What problems does it solve? Who would use it? Should be understandable by non-programmers. MAX 8-10 lines.]

**Technical Summary:**
[Technical details: Overall architecture, key components and their interactions, main technologies/frameworks used, API design patterns, how modules work together. For developers who need to understand the system design. MAX 8-10 lines.]

Guidelines:
- Focus on the REPOSITORY'S overall purpose and architecture, not individual modules
- Human summary: High-level purpose, problem domain, target users
- Technical summary: System architecture, key design patterns, technology stack, integration approach
- Explain how the different modules work together as a system
- Be concise but comprehensive

Example:
**Human Summary:** This is an HTTP client library for Python that makes it easy to send web requests and work with APIs. It simplifies common tasks like sending GET/POST requests, handling authentication, managing cookies, and working with HTTPS connections. The library is designed for developers who need a simple, intuitive way to interact with web services without dealing with low-level networking details. It's one of the most popular Python packages for making HTTP requests.

**Technical Summary:** Implements a high-level HTTP client built on top of urllib3 for connection pooling and low-level networking. Core architecture includes: Sessions for persistent configuration and connection reuse, Adapters for pluggable transport mechanisms with automatic retry logic, PreparedRequest/Response objects for request/response lifecycle management, and hook system for request/response processing. Supports connection pooling, SSL/TLS verification, proxy routing (HTTP/HTTPS/SOCKS), streaming uploads/downloads, and automatic content decoding. Uses urllib3's PoolManager for efficient connection reuse and provides simplified API surface over complex networking primitives.

Respond in the exact format shown above."""

        # Call API
        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=600, temperature=1)
        print(f"    Repository summary generated ({self.api_calls} API call)")

        return self._parse_two_level_summary(summary_text)

    def _summarize_large_repo(
        self,
        repo_name: str,
        module_summaries: Dict[str, Dict[str, str]],
        ast_summary: Dict
    ) -> Dict[str, str]:
        """
        Summarize very large repository using chunking

        """
        print(f"          Large repository ({len(module_summaries)} modules), using chunking")

        # Group modules by directory
        module_groups = self._group_modules_by_directory(module_summaries)

        print(f"         Analyzing {len(module_groups)} module groups")

        # Summarize each group
        group_summaries = []
        for idx, (group_name, modules) in enumerate(module_groups.items(), 1):
            modules_text = self._format_module_summaries(modules)

            prompt = f"""You are a senior Python engineer documenting one part of a larger codebase. 
This is group {idx}/{len(module_groups)} of a large Python repository — your description will later be combined with the other groups to produce the repository-level summary.

**Group:** {group_name}

**Modules in this group:**
{modules_text}

Describe what this GROUP of modules does in 2-3 sentences. Focus on their collective purpose and how they work together."""

            self.api_calls += 1
            group_summary = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=300, temperature=1)
            group_summaries.append(f"{group_name}: {group_summary}")
            print(f"         Group {idx}/{len(module_groups)} summarized")

        # Combine group summaries into overall repo summary
        repo_context = self._build_repo_context(repo_name, ast_summary)

        combined_prompt = f"""You are a senior Python engineer writing the README-level overview for a codebase you've just finished reviewing.
        
You analyzed a large Python repository `{repo_name}` in {len(module_groups)} groups, and the per-group summaries below are the only inputs you have. 
Another developer is about to start working in this repository for the first time, and your summary is the only context they will have before they open any file.

{repo_context}

**Module group summaries:**
{chr(10).join(f"{i}. {s}" for i, s in enumerate(group_summaries, 1))}

Treat the group summaries above as ground truth. Do not invent components, classes, or design patterns that aren't reflected there. Step back from the individual groups and ask: what is this 
  repository, taken as a whole? What category of software is it (library, framework, tool, application), and what's the central abstraction or workflow that all the groups support? Lead with that, and  
  treat the groups as the supporting cast — do not just enumerate them one by one.

Based on these group summaries, provide TWO summaries for the ENTIRE REPOSITORY:

**Human Summary:**
[Explain what this repository does in plain language. What is its overall purpose? MAX 8-10 lines.]

**Technical Summary:**
[Technical details: Overall architecture, key components, main technologies, how groups work together. MAX 8-10 lines.]

Keep both summaries concise and focused on the repository's overall role. Respond in the exact format shown above."""

        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": combined_prompt}], max_tokens=600, temperature=1)
        print(f"         Combined summary generated")

        return self._parse_two_level_summary(summary_text)

    def _group_modules_by_directory(
        self,
        module_summaries: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Group modules by their directory structure"""
        groups = {}

        for module_path, summary in module_summaries.items():
            # Extract directory 
            parts = module_path.split('/')
            if len(parts) > 1:
                group_name = '/'.join(parts[:-1])
            else:
                group_name = "root"

            if group_name not in groups:
                groups[group_name] = {}

            groups[group_name][module_path] = summary

        # If too many groups, merge smaller ones
        if len(groups) > self.MAX_MODULES_PER_CALL:
            # Keep top 30 groups, merge rest into "other"
            sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
            main_groups = dict(sorted_groups[:30])
            other_modules = {}
            for _, modules in sorted_groups[30:]:
                other_modules.update(modules)
            if other_modules:
                main_groups["other"] = other_modules
            return main_groups

        return groups

    def _build_repo_context(self, repo_name: str, ast_summary: Dict) -> str:
        """Build context string describing repository structure"""

        total_modules = ast_summary.get("total_modules", 0)
        total_functions = ast_summary.get("total_functions", 0)
        total_classes = ast_summary.get("total_classes", 0)

        context = f"""**Repository:** {repo_name}
**Total Modules:** {total_modules}
**Total Functions:** {total_functions}
**Total Classes:** {total_classes}"""

        
        return context

    def _format_module_summaries(
        self,
        module_summaries: Dict[str, Dict[str, str]]
    ) -> str:
        """Format module summaries into readable text"""

        lines = []
        for module_path, summary in module_summaries.items():
            human = summary.get('human', 'N/A')
            technical = summary.get('technical', 'N/A')

            lines.append(f" {module_path}:")
            lines.append(f"  - Purpose: {human}")
            lines.append(f"  - Technical: {technical}")
            lines.append("") 

        return '\n'.join(lines)

    def _parse_two_level_summary(self, summary_text: str) -> Dict[str, str]:
        """
        Parse LLM response into human and technical summaries

        """
        result = {"human": "", "technical": ""}

        lines = summary_text.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith('**Human Summary:**'):
                current_section = 'human'
                # Extract content after the header
                content = line.replace('**Human Summary:**', '').strip()
                if content:
                    result['human'] = content
            elif line.startswith('**Technical Summary:**'):
                current_section = 'technical'
                # Extract content after the header
                content = line.replace('**Technical Summary:**', '').strip()
                if content:
                    result['technical'] = content
            elif current_section and line:
                # Continuation of current section
                if result[current_section]:
                    result[current_section] += ' ' + line
                else:
                    result[current_section] = line

        return result


# Standalone CLI for testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m src.repo_summarizer <context_file.json>")
        print("\nExample:")
        print("  python -m src.repo_summarizer context_requests.json")
        sys.exit(1)

    context_file = sys.argv[1]

    print("="*80)
    print("REPOSITORY SUMMARIZER TEST")
    print("="*80)

    # Load context
    print(f"\n Loading context from {context_file}")
    with open(context_file, 'r') as f:
        context_data = json.load(f)

    repo_name = context_data.get('repo_name', 'Unknown')
    module_summaries = context_data.get('module_summaries', {})
    ast_results = context_data.get('ast_results', {})
    ast_summary = ast_results.get('__analysis_summary__', {})

    print(f"   Repo: {repo_name}")
    print(f"   Module summaries: {len(module_summaries)}")

    if not module_summaries:
        print("\n No module summaries found!")
        print("   Run analyze_any_repo.py with option 2 or 4 first.")
        sys.exit(1)

    # Generate repository summary
    print(f"\n  Generating repository summary")
    summarizer = RepoSummarizer()
    repo_summary = summarizer.summarize_repository(repo_name, module_summaries, ast_summary)

    print("\n" + "="*80)
    print(f"REPOSITORY SUMMARY")
    print("="*80)

    print(f"\n Human:")
    print(f"   {repo_summary.get('human', 'N/A')}")

    print(f"\n Technical:")
    print(f"   {repo_summary.get('technical', 'N/A')}")

    # Save to context
    context_data['repo_summary'] = repo_summary
    with open(context_file, 'w') as f:
        json.dump(context_data, f, indent=2)

    print(f"\n Repository summary saved to: {context_file}")
    print(f" Total API calls: {summarizer.api_calls}")
