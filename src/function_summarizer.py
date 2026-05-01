"""
Function Summarizer - Generate concise summaries using LLM
Simple strategy: One function at a time, with chunking for very large functions
"""

import json
from typing import Dict, List, Optional
import config


class FunctionSummarizer:
    """
    Generate summaries for Python functions 

    """

    # Character limits
    MAX_FUNCTION_CHARS = 12000  # ~200-250 lines
    CHUNK_LINES = 60  # Lines per chunk for very large functions
    CHUNK_OVERLAP = 15  # Overlap between chunks for context continuity

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with TAMU AI Chat API key"""
        self.client = config.get_client()
        self.model = config.LLM_MODEL

        # Track API calls for monitoring
        self.api_calls = 0

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

    def summarize_module_functions(
        self,
        module_path: str,
        functions: List[Dict]
    ) -> Dict[str, Dict[str, str]]:
        """
        Summarize all functions in a module that need LLM summaries
        """
        # Filter functions that need summarization
        to_summarize = [
            f for f in functions
            if f.get('needs_llm_summary', False)
        ]

        if not to_summarize:
            print(f"    No functions need summarization in {module_path}")
            return {}

        print(f"\n    Summarizing {len(to_summarize)} functions in {module_path}")

        summaries = {}

        for idx, func in enumerate(to_summarize, 1):
            func_name = func['name']
            line_count = func.get('line_count', 0)

            print(f"      [{idx}/{len(to_summarize)}] {func_name} ({line_count} lines)", end=" ")

            try:
                summary = self.summarize_function(func, module_path)
                summaries[func_name] = summary
                

            except Exception as e:
                print(f"⚠️  Error: {e}")
                summaries[func_name] = {
                    "human": f"Function: {func_name}",
                    "technical": f"Function with {line_count} lines (summary generation failed)"
                }

        print(f"    Generated {len(summaries)} summaries ({self.api_calls} API calls)")
        return summaries

    def summarize_function(self, func: Dict, module_path: str) -> str:
        """
        Summarize a single function

        Decides whether to use full code or chunking based on size
        """
        code = func.get('code', '')
        code_size = len(code)

        # Small/medium function - send full code
        if code_size <= self.MAX_FUNCTION_CHARS:
            return self._summarize_full_function(func)

        # Very large function - use chunking
        print(f"\n          Large function ({code_size} chars), using chunking")
        return self._summarize_with_chunking(func)

    def _summarize_full_function(self, func: Dict) -> str:
        """Summarize function by sending full code to LLM"""

        func_name = func['name']
        sig = self._format_signature(func)
        code = func.get('code', '')
        line_count = func.get('line_count', 0)
        docstring = func.get('docstring', '')
        calls = func.get('calls', [])[:20]
        decorators = func.get('decorators', [])

        # Build prompt
        prompt = f"""You are a senior Python engineer documenting an unfamiliar codebase. 
    
    Your goal is to write summaries that help another developer understand this function quickly and accurately.    
    Analyze this Python function and provide a TWO-LEVEL summary.

**Function:** {func_name}
**Signature:** {sig}
**Lines:** {line_count}
**Decorators:** {', '.join([d['name'] for d in decorators]) if decorators else 'None'}

{f"**Existing docstring:** {docstring}" if docstring else "**No docstring provided**"}

**Key function calls:** {', '.join(calls) if calls else 'None'}

**Full code:**
```python
{code}
```

---
Ground every claim in what the code actually does. Do not infer behavior from the function name alone. If something is unclear from the code, omit it rather than guess.
Provide TWO summaries:

**Human Summary:**
[Explain what this function does in plain, simple language. Focus on the PURPOSE and what problem it solves. Should be understandable by non-programmers. MAX 5-6 lines.]

**Technical Summary:**
[Provide technical details: key operations, data flow, important APIs or dependencies, what it returns, side effects. For developers who need to understand implementation. MAX 5-6 lines.]

Guidelines:
- Don't just repeat the docstring - analyze the actual code
- Human summary: Plain English, focus on WHAT and WHY
- Technical summary: Specific operations, mention important function calls, data transformations
- Keep BOTH summaries concise: maximum 5-6 lines each
- Be clear and complete but avoid unnecessary verbosity

Example:
**Human Summary:** Connects to the email server so the assistant can send and receive emails. Sets up the communication channel needed for email operations.

**Technical Summary:** Establishes async connection to MCP email server via stdio by launching email_server.py subprocess. Creates ClientSession with bidirectional streams, initializes session handshake, and stores session/context as instance attributes for subsequent operations.

Respond in the exact format shown above."""

        # Call API
        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=400, temperature=1)
        return self._parse_two_level_summary(summary_text)

    def _extract_function_context(self, func: Dict) -> str:
        """
        Extract key context from full function for chunk summarization

        """
        sig = self._format_signature(func)
        line_count = func.get('line_count', 0)
        calls = func.get('calls', [])[:15]
        decorators = [d['name'] for d in func.get('decorators', [])]

        # Extract variable patterns from code
        code = func.get('code', '')
        import re
        # Find assignments
        assignments = re.findall(r'(\w+)\s*=', code)
        # Deduplicate and limit
        variables = list(dict.fromkeys(assignments))[:20]

        context = f"""FUNCTION CONTEXT:
Signature: {sig}
Total Lines: {line_count}
Decorators: {', '.join(decorators) if decorators else 'None'}
Key Variables: {', '.join(variables) if variables else 'None'}
Key Operations: {', '.join(calls) if calls else 'None'}"""

        return context

    def _summarize_with_chunking(self, func: Dict) -> Dict[str, str]:
        """
        Summarize very large function using chunking strategy with context injection

        Process:
        1. Extract function context (variables, calls, signature)
        2. Split function into OVERLAPPING chunks for continuity
        3. Summarize each chunk WITH context
        4. Combine chunk summaries into overall summary

        """
        code = func.get('code', '')
        lines = code.split('\n')
        func_name = func['name']

        # Extract context for all chunks
        context_info = self._extract_function_context(func)

        # Split into OVERLAPPING chunks
        chunks = []
        i = 0
        while i < len(lines):
            chunk_end = min(i + self.CHUNK_LINES, len(lines))
            chunk = '\n'.join(lines[i:chunk_end])
            chunks.append({
                'code': chunk,
                'start_line': i + 1,
                'end_line': chunk_end
            })
            # Move forward by (CHUNK_LINES - OVERLAP) for overlapping
            i += (self.CHUNK_LINES - self.CHUNK_OVERLAP)
          
            if chunk_end >= len(lines):
                break

        print(f"         Analyzing {len(chunks)} overlapping chunks")

        # Step 1: Summarize each chunk WITH CONTEXT
        chunk_summaries = []
        for idx, chunk in enumerate(chunks, 1):
            prompt = f"""You are a senior Python engineer documenting an unfamiliar codebase. 
            You are reading one section of a large function so you can later combine all sections into a coherent summary.

{context_info}

**CHUNK {idx}/{len(chunks)} (lines {chunk['start_line']}-{chunk['end_line']}):**
```python
{chunk['code']}
```

Given the FULL FUNCTION CONTEXT above (variables, operations, signature), describe what THIS CHUNK does in 1-2 sentences.
Focus on the logic in this section, knowing the variables and calls exist in the full function."""

            self.api_calls += 1
            chunk_summary = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=300, temperature=1)
            chunk_summaries.append(chunk_summary)
            print(f"         Chunk {idx}/{len(chunks)} summarized.")

        # Step 2: Combine chunk summaries into overall summary
        combined_prompt = f"""You analyzed a large Python function `{func_name}` in {len(chunks)} overlapping sections. Now provide a TWO-LEVEL overall summary.

{context_info}

**Section summaries (with overlap for continuity):**
{chr(10).join(f"{i}. {s}" for i, s in enumerate(chunk_summaries, 1))}
Ground every claim in the section summaries below. Do not invent steps that weren't described. If sections overlap, treat them as the same logic, not separate steps.
Based on these section summaries, provide TWO summaries:

**Human Summary:**
[Explain what this function does in plain language - the overall purpose and what it accomplishes. MAX 5-6 lines.]

**Technical Summary:**
[Technical details: key operations, workflow steps, what it returns, important dependencies. MAX 5-6 lines.]

Keep both summaries concise and focused. Respond in the exact format shown above."""

        self.api_calls += 1
        summary_text = config.llm_call(self.client, self.model, [{"role": "user", "content": combined_prompt}], max_tokens=400, temperature=1)
        print(f"         Combined summary generated.")

        return self._parse_two_level_summary(summary_text)

    def _format_signature(self, func: Dict) -> str:
        """Format function signature with types"""
        name = func['name']
        args = func.get('args', [])

        # Format arguments (skip 'self')
        args_str = ', '.join([
            f"{arg['name']}: {arg.get('type', 'Any')}"
            for arg in args
            if arg['name'] != 'self'
        ])

        return_type = func.get('return_type', 'None')
        async_prefix = 'async ' if func.get('is_async') else ''

        return f"{async_prefix}def {name}({args_str}) -> {return_type}"

    def get_function_description(
        self,
        func: Dict,
        summary: Optional[str] = None
    ) -> str:
        """
        Get best available description for a function
        """
        # Use provided summary
        if summary:
            return summary

        # Small function - code is self-documenting
        if not func.get('needs_llm_summary'):
            code = func.get('code', '')
            if code:
                return f"```python\n{code}\n```"

        # Use docstring if available
        if func.get('docstring'):
            return func['docstring']

        # Fallback to signature
        return self._format_signature(func)


# Standalone CLI for testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.ast_parser import CodeAnalyzer

    if len(sys.argv) < 2:
        print("Usage: python -m src.function_summarizer <path_to_repo>")
        print("\nExample:")
        print("  python -m src.function_summarizer ../test_project")
        sys.exit(1)

    repo_path = sys.argv[1]

    print("="*80)
    print("FUNCTION SUMMARIZER TEST")
    print("="*80)

    print("\n Step 1: Running AST analysis")
    analyzer = CodeAnalyzer(repo_path)
    results = analyzer.analyze_repository()

    print("\n Step 2: Generating function summaries")
    summarizer = FunctionSummarizer()

    all_summaries = {}
    total_functions = 0

    for module_path, module_data in results.items():
        if module_path == '__analysis_summary__' or 'error' in module_data:
            continue

        functions = module_data.get('functions', [])
        if not functions:
            continue

        summaries = summarizer.summarize_module_functions(module_path, functions)
        if summaries:
            all_summaries[module_path] = summaries
            total_functions += len(summaries)

    print("\n" + "="*80)
    print(f"SUMMARY RESULTS")
    print("="*80)
    print(f"Total functions summarized: {total_functions}")
    print(f"Total API calls: {summarizer.api_calls}")
    print(f"Modules processed: {len(all_summaries)}")

    print("\n" + "="*80)
    print("GENERATED SUMMARIES")
    print("="*80)

    for module_path, summaries in all_summaries.items():
        print(f"\n {module_path}:")
        for func_name, summary_dict in summaries.items():
            print(f"\n   {func_name}:")
            print(f"     Simple: {summary_dict.get('human', 'N/A')}")
            print(f"     Technical: {summary_dict.get('technical', 'N/A')}")

    # Save to JSON
    output_file = f"function_summaries_{Path(repo_path).name}.json"
    with open(output_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n Summaries saved to: {output_file}")
