"""
Module Summarizer - Generate module-level summaries from function summaries
Aggregates all function summaries in a Python file to create a cohesive module summary
"""

import json
from typing import Dict, List, Optional
import config


class ModuleSummarizer:
    """
    Generate summaries for Python modules (files) using function summaries

    """

   
    MAX_FUNCTIONS_PER_CALL = 30  # Max functions to include in one API call

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with TAMU AI Chat API key"""
        self.client = config.get_client()
        self.model = config.LLM_MODEL

        # Track API calls
        self.api_calls = 0

    def summarize_all_modules(
        self,
        ast_results: Dict,
        function_summaries: Dict[str, Dict[str, Dict[str, str]]]
    ) -> Dict[str, Dict[str, str]]:
        """
        Summarize all modules in a repository

        """
        module_summaries = {}

        print(f"\n Generating module summaries")

        for module_path, module_data in ast_results.items():
          
            if module_path == '__analysis_summary__' or 'error' in module_data:
                continue

            # Get function summaries for this module
            func_summaries = function_summaries.get(module_path, {})

            print(f"    {module_path}", end=" ")

            try:
                summary = self.summarize_module(module_path, module_data, func_summaries)
                module_summaries[module_path] = summary
                

            except Exception as e:
                print(f"  Error: {e}")
                module_summaries[module_path] = {
                    "human": f"Module: {module_path}",
                    "technical": f"Module analysis failed: {str(e)}"
                }

        print(f"    Generated {len(module_summaries)} module summaries ({self.api_calls} API calls)")
        return module_summaries

    def summarize_module(
        self,
        module_path: str,
        module_data: Dict,
        function_summaries: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Summarize a single module

        """
        # Extract module metadata
        classes = module_data.get('classes', [])
        imports = module_data.get('imports', [])
        functions = module_data.get('functions', [])

        # If no functions/classes, summarize from source code
        if not function_summaries and not functions and not classes:
            return self._summarize_module_from_code(module_path, module_data)

        # Check if we need to chunk (large module with many functions)
        if len(function_summaries) > self.MAX_FUNCTIONS_PER_CALL:
            return self._summarize_large_module(
                module_path, module_data, function_summaries
            )

        # Build module context
        module_context = self._build_module_context(
            module_path, classes, imports, functions
        )

        # Build function summaries text
        func_summaries_text = self._format_function_summaries(function_summaries)

        # Generate prompt
        prompt = f"""You are a senior Python engineer documenting an unfamiliar codebase. 
Your goal is to write a module-level summary that helps another developer understand the module's role at a glance, without forcing them to read every function.
        
Analyze this Python module and provide a TWO-LEVEL summary.

{module_context}

**Function Summaries:**
{func_summaries_text}

---
Ground every claim in the structure and function summaries above. 
Do not invent classes, APIs, or design patterns that weren't described. 
Describe the module's overall role — do not just list its functions one by one.

Imagine the reader has never seen this codebase before. They should finish reading your summary knowing what this module is for, when they would touch it, and what it would mean to remove it.

Based on the module structure and function summaries above, provide TWO summaries:

**Human Summary:**
[Explain what this module does in plain language. What is its purpose in the codebase? What problems does it solve? Should be understandable by non-programmers. MAX 6-8 lines.]

**Technical Summary:**
[Technical details: Key classes and their responsibilities, main APIs/interfaces provided, important dependencies, design patterns used, how it fits in the larger system. For developers who need to understand the module architecture. MAX 6-8 lines.]

Guidelines:
- Focus on the MODULE'S overall purpose and role, not individual functions
- Human summary: High-level purpose, what problem domain it addresses
- Technical summary: Architecture, key classes, API surface, integration points
- Be concise but complete

Example:
**Human Summary:** This module handles all HTTP connection management for the requests library. It manages connection pools to reuse connections efficiently, handles SSL certificate verification for secure connections, and provides proxy server support. It's the core networking layer that actually sends requests over the internet and manages the underlying socket connections.

**Technical Summary:** Implements HTTPAdapter class that wraps urllib3's PoolManager for connection pooling. Provides send() method as primary interface for executing PreparedRequest objects. Manages SSL/TLS context via cert_verify(), handles proxy routing through proxy_manager_for(), and implements connection pool key generation for connection reuse. Converts urllib3 exceptions to requests-library exception types and builds Response objects from raw urllib3 responses.

Respond in the exact format shown above."""

        # Call API
        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=500, temperature=1)
        return self._parse_two_level_summary(summary_text)

    def _summarize_large_module(
        self,
        module_path: str,
        module_data: Dict,
        function_summaries: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Summarize a module with many functions using chunking

        """
        print(f"\n           Large module ({len(function_summaries)} functions), using chunking")

        # Split functions into chunks
        func_items = list(function_summaries.items())
        chunks = []

        for i in range(0, len(func_items), self.MAX_FUNCTIONS_PER_CALL):
            chunk = dict(func_items[i:i + self.MAX_FUNCTIONS_PER_CALL])
            chunks.append(chunk)

        print(f"         Analyzing {len(chunks)} function groups")

        # Summarize each chunk
        chunk_summaries = []
        for idx, chunk in enumerate(chunks, 1):
            func_summaries_text = self._format_function_summaries(chunk)

            prompt = f"""This is part {idx}/{len(chunks)} of a large Python module.

**Functions in this section:**
{func_summaries_text}

Describe what this GROUP of functions does in 2-3 sentences. Focus on their collective purpose and what functionality they provide together."""

            self.api_calls += 1
            chunk_summary = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=300, temperature=1)
            chunk_summaries.append(chunk_summary)
            print(f"         Chunk {idx}/{len(chunks)} summarized")

        # Combine chunk summaries into overall module summary
        module_context = self._build_module_context(
            module_path,
            module_data.get('classes', []),
            module_data.get('imports', []),
            module_data.get('functions', [])
        )

        combined_prompt = f"""You are a senior Python engineer documenting an unfamiliar codebase.
        You analyzed a large Python module `{module_path}` in {len(chunks)} parts.

{module_context}

**Function group summaries:**
{chr(10).join(f"{i}. {s}" for i, s in enumerate(chunk_summaries, 1))}

Ground every claim below in the function group summaries above. 
Do not invent classes, APIs, or design patterns that weren't described. 
If two groups describe overlapping responsibilities, treat them as one capability rather than listing both. 

Based on these function groups, provide TWO summaries for the ENTIRE MODULE:

**Human Summary:**
[Explain what this module does in plain language. What is its overall purpose? MAX 6-8 lines.]

**Technical Summary:**
[Technical details: Key classes, main APIs, important dependencies, design patterns. MAX 6-8 lines.]

Keep both summaries concise and focused on the module's overall role. Respond in the exact format shown above."""

        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": combined_prompt}], max_tokens=500, temperature=1)
        print(f"         Combined summary generated")

        return self._parse_two_level_summary(summary_text)

    def _summarize_module_from_code(
        self,
        module_path: str,
        module_data: Dict
    ) -> Dict[str, str]:
        """
        Summarize a module that has no functions/classes by reading its source code
        """
        # Get source code preview from AST data
        code_preview = module_data.get('code_preview', '')
        line_count = module_data.get('line_count', 0)
        imports = module_data.get('imports', [])

        # If module is essentially empty (1 line or less), return simple summary
        if line_count <= 1:
            return {
                "human": f"This is an empty or placeholder module file.",
                "technical": f"Module '{module_path}' contains no significant code (empty file or single comment/blank line)."
            }

        # Build import context
        import_list = []
        for imp in imports[:15]:
            if imp.get('module'):
                import_list.append(imp['module'])

        # Build prompt with code preview
        prompt = f"""You are a senior Python engineer documenting an unfamiliar codebase. 
        This module contains no functions or classes — only imports, constants, type definitions, or other top-level 
  statements. Your job is to describe what the module actually contains, based only on the code below.

**Module:** {module_path}
**Lines of code:** {line_count}
**Imports:** {', '.join(import_list) if import_list else 'None'}

**Code Preview:**
```python
{code_preview[:1000] if code_preview else 'No code preview available'}
```

Ground every claim in the code shown above. 
Do not assume the module is a "configuration" file or a "settings" module unless the code clearly shows that — many empty-of-functions modules are package  
initializers, type aliases, version markers, or public re-exports. Describe what is actually there.

Provide TWO summaries:

**Human Summary:**
[Explain what this module provides in plain language. What is its purpose? What does it export or define? MAX 4-5 lines.]

**Technical Summary:**
[Technical details: What constants/variables are defined, what modules are imported, how this fits in the system architecture. MAX 4-5 lines.]

Guidelines:
- Focus on what the module PROVIDES (constants, types, imports)
- Explain its role in the larger codebase
- Be concise

Example:
**Human Summary:** This module serves as a configuration file that defines constants and settings used throughout the application. It imports necessary dependencies and sets up global variables that other modules rely on for consistent behavior.

**Technical Summary:** Defines configuration constants including API endpoints, timeout values, and feature flags. Imports third-party dependencies (requests, json) and sets up module-level variables for connection pooling parameters. Acts as central configuration point referenced by api.py and utils.py modules.

Respond in the exact format shown above."""

        # Call API
        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=400, temperature=1)
        return self._parse_two_level_summary(summary_text)

    def _build_module_context(
        self,
        module_path: str,
        classes: List[Dict],
        imports: List[Dict],
        functions: List[Dict]
    ) -> str:
        """Build context string describing module structure"""

        # Extract class names
        class_names = [cls['name'] for cls in classes]

        # Extract key imports (limit to top 10)
        import_names = []
        for imp in imports[:10]:
            if imp.get('module'):
                import_names.append(imp['module'])

        # Count functions
        func_count = len(functions)

        context = f"""**Module:** {module_path}
**Classes:** {', '.join(class_names) if class_names else 'None'}
**Key Imports:** {', '.join(import_names) if import_names else 'None'}
**Function Count:** {func_count}"""

        return context

    def _format_function_summaries(
        self,
        function_summaries: Dict[str, Dict[str, str]]
    ) -> str:
        """Format function summaries into readable text"""

        lines = []
        for func_name, summaries in function_summaries.items():
            human = summaries.get('human', 'N/A')
            technical = summaries.get('technical', 'N/A')

            lines.append(f"• {func_name}:")
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
        print("Usage: python -m src.module_summarizer <context_file.json>")
        print("\nExample:")
        print("  python -m src.module_summarizer context_requests.json")
        sys.exit(1)

    context_file = sys.argv[1]

    print("="*80)
    print("MODULE SUMMARIZER TEST")
    print("="*80)

    # Load context
    print(f"\n Loading context from {context_file}")
    with open(context_file, 'r') as f:
        context_data = json.load(f)

    ast_results = context_data.get('ast_results', {})
    function_summaries = context_data.get('function_summaries', {})

    print(f"   Modules: {len([k for k in ast_results.keys() if k != '__analysis_summary__'])}")
    print(f"   Function summaries: {sum(len(v) for v in function_summaries.values())}")

    # Generate module summaries
    print(f"\n Generating module summaries")
    summarizer = ModuleSummarizer()
    module_summaries = summarizer.summarize_all_modules(ast_results, function_summaries)

    print("\n" + "="*80)
    print(f"MODULE SUMMARIES")
    print("="*80)

    for module_path, summary in module_summaries.items():
        print(f"\n {module_path}:")
        print(f"\n   Human:")
        print(f"     {summary.get('human', 'N/A')}")
        print(f"\n   Technical:")
        print(f"     {summary.get('technical', 'N/A')}")

    # Save to JSON
    output_file = f"module_summaries_{Path(context_file).stem.replace('context_', '')}.json"
    with open(output_file, 'w') as f:
        json.dump(module_summaries, f, indent=2)

    print(f"\n Module summaries saved to: {output_file}")
    print(f"Total API calls: {summarizer.api_calls}")
