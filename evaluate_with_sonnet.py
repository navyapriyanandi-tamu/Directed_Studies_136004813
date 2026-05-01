#!/usr/bin/env python3
"""
Evaluate CodeChat outputs using Claude Sonnet 4.5 as a judge.

Pipeline:
  1. Load context_<repo>.json (produced by analyze_any_repo.py with Haiku)
  2. Use Claude Sonnet 4.5 (via TAMU API) to score each component on a 1-5 rubric
  3. Save results to evaluation_<repo>.json and print a summary

Usage:
    python3 evaluate_with_sonnet.py context_httpx.json
    python3 evaluate_with_sonnet.py httpx
"""

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import config

# ---------------- Sampling configuration ----------------
# Function summaries are evaluated using stratified sampling across modules
TARGET_FUNCTION_SAMPLE_SIZE = 50   # Total functions evaluated per repo
SKIP_DUNDER_METHODS = True         # Skip __init__, __repr__, etc.
MIN_FUNCTION_LINES = 5             # Skip functions shorter than this
PREFER_LONGER_FUNCTIONS = True     # Within a module, weight by length
SAMPLE_RANDOM_SEED = 42            # Reproducibility


class OpusEvaluator:
    """Uses Claude Opus 4.5 to evaluate Haiku-generated outputs."""

    def __init__(self):
        self.client = config.get_client()
        self.judge_model = config.JUDGE_MODEL

    def _call_with_retry(self, prompt: str, max_tokens: int = 2000, max_retries: int = 5) -> str:
        """Call the judge model with exponential backoff on transient failures."""
        for attempt in range(max_retries):
            try:
                return config.llm_call(
                    self.client,
                    self.judge_model,
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=1,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + 1
                    print(f"   API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"   API failed after {max_retries} attempts: {e}")
                    raise

    def _safe_json_parse(self, response_text: str, default: Dict = None) -> Dict:
        """Parse JSON from a response"""
        text = response_text.strip()
        if text.startswith("```json"):
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"   JSON parse error: {e}. Trying to fix escapes")
            fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', text)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as e2:
                print(f"   Still failed: {e2}")
                print(f"   Response (first 500 chars): {text[:500]}")
                if default is not None:
                    print(f"   Using default response")
                    return default
                raise

    # ---------------- Function summaries ----------------

    def evaluate_function_summary(self, function_code: str, summary: Dict) -> Dict:
        """Score a single function summary against its source code."""
        MAX_CODE_LENGTH = 8000
        if len(function_code) > MAX_CODE_LENGTH:
            lines = function_code.split('\n')
            chunks, i = [], 0
            CHUNK_LINES, CHUNK_OVERLAP = 60, 15
            while i < len(lines):
                end = min(i + CHUNK_LINES, len(lines))
                chunks.append((i + 1, end, '\n'.join(lines[i:end])))
                if end >= len(lines):
                    break
                i += (CHUNK_LINES - CHUNK_OVERLAP)
            code_display = ""
            for idx, (start, end, code) in enumerate(chunks, 1):
                code_display += f"\n--- CHUNK {idx}/{len(chunks)} (lines {start}-{end}, overlapping) ---\n{code}\n"
        else:
            code_display = function_code

        prompt = f"""You are a strict but fair code-review evaluator scoring an AI-generated summary of a Python function. 

Your goal is to produce ratings that reliably distinguish good summaries from poor ones — not to be polite.
        
CODE:
```python
{code_display}
```

GENERATED SUMMARY:
Human: {summary.get('human', 'N/A')}
Technical: {summary.get('technical', 'N/A')}

Use the full 1-5 scale. A 5 should be reserved for summaries with no factual errors and no meaningful gaps, a 3 means clearly usable but with notable issues,
a 1-2 means the summary would mislead a developer. Do not default to 4 or 5 if the summary has identifiable problems.

Rate each criterion 1-5:

  1. FACTUAL ACCURACY: Does the summary correctly describe what the code does? Any hallucinations? Cite the specific line or behavior in the code if you find an error.                                   
  2. COMPLETENESS: Does it cover main functionality, return values, and side effects? List any concrete element from the code that the summary missed.                                                  
  3. CLARITY: Is it understandable to its target audience (the human summary for general readers, the technical summary for developers)?                                                                  
  4. TECHNICAL DEPTH: Is the technical summary appropriately detailed — neither too shallow (just paraphrasing the human summary) nor too verbose?

Respond in JSON:
{{
    "factual_accuracy": {{"rating": <1-5>, "errors_found": [], "notes": ""}},
    "completeness": {{"rating": <1-5>, "missing_elements": [], "notes": ""}},
    "clarity": {{"rating": <1-5>, "notes": ""}},
    "technical_depth": {{"rating": <1-5>, "notes": ""}},
    "overall_score": <1-5>,
    "recommendation": "Accept as-is|Minor issues|Major issues",
    "suggested_improvements": ""
}}"""

        default = {
            "factual_accuracy": {"rating": 3, "errors_found": [], "notes": "Eval failed"},
            "completeness": {"rating": 3, "missing_elements": [], "notes": "Eval failed"},
            "clarity": {"rating": 3, "notes": "Eval failed"},
            "technical_depth": {"rating": 3, "notes": "Eval failed"},
            "overall_score": 3,
            "recommendation": "Minor issues",
            "suggested_improvements": "JSON parsing error",
        }
        return self._safe_json_parse(self._call_with_retry(prompt, max_tokens=2000), default)

    # ---------------- Module summaries ----------------

    def evaluate_module_summaries(
        self,
        ast_results: Dict,
        module_summaries: Dict,
        function_summaries: Dict,
    ) -> Dict:
        """Score every module-level summary."""
        evaluations = {}
        for module_name, summary in module_summaries.items():
            print(f"   Module: {module_name}")
            module_ast = ast_results.get(module_name, {})
            funcs = module_ast.get('functions', [])
            classes = module_ast.get('classes', [])
            imports = module_ast.get('imports', [])

            mod_func_summaries = function_summaries.get(module_name, {})
            fn_text = ""
            for fn_name, fn_sum in mod_func_summaries.items():
                if isinstance(fn_sum, dict):
                    fn_text += (
                        f"\n\n  {fn_name}():\n"
                        f"    Human: {fn_sum.get('human', 'N/A')}\n"
                        f"    Technical: {fn_sum.get('technical', 'N/A')}"
                    )

            imports_str = ', '.join(
                imp.get('module', str(imp)) if isinstance(imp, dict) else str(imp)
                for imp in imports
            ) or 'None'

            prompt = f"""You are a strict but fair code-review evaluator scoring an AI-generated summary of a Python module. 
            Your goal is to produce ratings that reliably distinguish good module summaries from poor ones — not to be polite.                                                                                                                                                            
   
MODULE: {module_name}

STRUCTURE:
- Functions ({len(funcs)}): {', '.join(f['name'] for f in funcs) or 'None'}
- Classes ({len(classes)}): {', '.join(c['name'] for c in classes) or 'None'}
- Imports: {imports_str}

FUNCTION SUMMARIES (used to build this module summary):
{fn_text or "  (No function summaries)"}

GENERATED MODULE SUMMARY:
Human: {summary.get('human', 'N/A')}
Technical: {summary.get('technical', 'N/A')}

Use the full 1-5 scale. A 5 should be reserved for summaries with no factual errors and no meaningful gaps, a 3 means clearly usable but with notable issues, a 1-2 means the summary would mislead a 
developer. Do not default to 4 or 5 if the summary has identifiable problems.

Rate 1-5:
  1. ACCURACY: Correctly describes what this module does, grounded in the structure and function summaries above? Cite any specific function or class the summary describes incorrectly.                
  2. COMPLETENESS: Does the summary capture the module's main responsibilities, or does it miss capabilities visible in the function summaries? List any concrete element it missed.                      
  3. CLARITY: Easy to understand for the target audience (the human summary for general readers, the technical summary for developers)?

Respond in JSON:
{{
    "accuracy": {{"rating": <1-5>, "notes": ""}},
    "completeness": {{"rating": <1-5>, "notes": ""}},
    "clarity": {{"rating": <1-5>, "notes": ""}},
    "overall_score": <1-5>
}}"""

            default = {
                "accuracy": {"rating": 3, "notes": "Eval failed"},
                "completeness": {"rating": 3, "notes": "Eval failed"},
                "clarity": {"rating": 3, "notes": "Eval failed"},
                "overall_score": 3,
            }
            evaluations[module_name] = self._safe_json_parse(
                self._call_with_retry(prompt, max_tokens=1500), default
            )

        avg = (
            round(sum(e['overall_score'] for e in evaluations.values()) / len(evaluations), 2)
            if evaluations
            else 0
        )
        return {
            "modules_evaluated": list(evaluations.keys()),
            "individual_evaluations": evaluations,
            "average_score": avg,
        }

    # ---------------- Repo summary ----------------

    def evaluate_repo_summary(
        self,
        ast_results: Dict,
        repo_summary: Dict,
        module_summaries: Dict,
    ) -> Dict:
        """Score the repository-level summary."""
        modules = [k for k in ast_results.keys() if k != '__analysis_summary__']
        total_funcs = sum(
            len(m.get('functions', [])) for m in ast_results.values() if isinstance(m, dict)
        )
        total_classes = sum(
            len(m.get('classes', [])) for m in ast_results.values() if isinstance(m, dict)
        )

        mod_text = ""
        for mod, s in module_summaries.items():
            mod_text += (
                f"\n\n{mod}:\n"
                f"  Human: {s.get('human', 'N/A')}\n"
                f"  Technical: {s.get('technical', 'N/A')}"
            )

        prompt = f"""You are a strict but fair code-review evaluator scoring an AI-generated summary of a Python repository. Your goal is to produce ratings that reliably distinguish good repository  
  summaries from poor ones — not to be polite.
        
REPOSITORY:
- Modules: {len(modules)}
- Functions: {total_funcs}
- Classes: {total_classes}
- Files: {', '.join(modules)}

MODULE SUMMARIES (used to build this repo summary):
{mod_text}

GENERATED REPOSITORY SUMMARY:
Human: {repo_summary.get('human', 'N/A')}
Technical: {repo_summary.get('technical', 'N/A')}

Use the full 1-5 scale. A 5 should be reserved for summaries with no factual errors and no meaningful gaps, a 3 means clearly usable but with notable issues, a 1-2 means the summary would mislead a 
developer. Do not default to 4 or 5 if the summary has identifiable problems.

Rate 1-5:
  1. ACCURACY: Correctly describes the repository's purpose and structure, grounded in the module summaries above? Cite any specific component the summary describes incorrectly.                       
  2. COMPLETENESS: Covers main functionality, key components, and overall architecture? List any major capability visible in the module summaries that the repository summary missed.                     
  3. CLARITY: Understandable for someone new to the codebase, with the human summary aimed at general readers and the technical summary aimed at developers?                                              
  4. USEFULNESS: Would this help a developer onboard quickly — does it tell them what the repository is, when they would touch it, and what would be lost without it? 

Respond in JSON:
{{
    "accuracy": {{"rating": <1-5>, "errors_found": [], "notes": ""}},
    "completeness": {{"rating": <1-5>, "missing_elements": [], "notes": ""}},
    "clarity": {{"rating": <1-5>, "notes": ""}},
    "usefulness": {{"rating": <1-5>, "notes": ""}},
    "overall_score": <1-5>,
    "strengths": [],
    "weaknesses": []
}}"""

        default = {
            "accuracy": {"rating": 3, "errors_found": [], "notes": "Eval failed"},
            "completeness": {"rating": 3, "missing_elements": [], "notes": "Eval failed"},
            "clarity": {"rating": 3, "notes": "Eval failed"},
            "usefulness": {"rating": 3, "notes": "Eval failed"},
            "overall_score": 3,
            "strengths": [],
            "weaknesses": [],
        }
        return self._safe_json_parse(self._call_with_retry(prompt, max_tokens=3000), default)

    # ---------------- Architecture diagram ----------------

    def evaluate_architecture_diagram(
        self,
        files: List[str],
        mermaid_code: str,
        logical_groups: Dict,
    ) -> Dict:
        """Score the Mermaid architecture diagram and logical groupings."""
        prompt = f"""You are a strict but fair architecture reviewer scoring an AI-generated architecture diagram for a Python repository. 
        Your goal is to produce ratings that reliably distinguish clear, useful diagrams from cluttered or misleading ones — not to be polite. 

REPOSITORY FILES ({len(files)} total):
{', '.join(files)}

MERMAID DIAGRAM:
```mermaid
{mermaid_code}
```

LOGICAL GROUPS:
{json.dumps(logical_groups, indent=2)}

Use the full 1-5 scale. A 5 should be reserved for diagrams with no missing components, semantically clean groupings, and a clear, readable layout, a 3 means usable but with notable issues, a 1-2     
means the diagram would mislead a developer about the architecture. Do not default to 4 or 5 if the diagram has identifiable problems.

Rate 1-5:
  1. COMPLETENESS: Are all important files represented in the logical groups, and does the diagram itself surface them? List any major file that is missing from the diagram or buried inside a generic
  "Other" bucket.                                                                                                                                                                                         
  2. LOGICAL GROUPING: Do the group names and memberships reflect actual architectural roles (e.g., "Transport Layer", "Authentication") rather than generic or directory-shaped buckets? Call out any
  module that is grouped incorrectly.                                                                                                                                                                     
  3. DIAGRAM QUALITY: Is the rendered Mermaid diagram readable and informative — clear hierarchy, no overwhelming edge density, no nonsensical cycles, no architecturally inverted arrows?
  Flag specific problems if found.

Respond in JSON:
{{
    "completeness": {{"rating": <1-5>, "files_in_repo": {len(files)}, "notes": ""}},
    "logical_grouping": {{"rating": <1-5>, "notes": ""}},
    "diagram_quality": {{"rating": <1-5>, "notes": ""}},
    "overall_score": <1-5>,
    "suggestions": []
}}"""

        default = {
            "completeness": {"rating": 3, "files_in_repo": len(files), "notes": "Eval failed"},
            "logical_grouping": {"rating": 3, "notes": "Eval failed"},
            "diagram_quality": {"rating": 3, "notes": "Eval failed"},
            "overall_score": 3,
            "suggestions": [],
        }
        return self._safe_json_parse(self._call_with_retry(prompt, max_tokens=2000), default)

    # ---------------- Stratified function sampling ----------------

    def _sample_functions(
        self,
        function_summaries: Dict[str, Dict],
        ast_results: Dict,
        target_size: int = TARGET_FUNCTION_SAMPLE_SIZE,
    ) -> List[Tuple[str, str, Dict, str]]:
        """Pick a structurally representative subset of functions to evaluate.

        Returns a list of (module, fn_name, summary, fn_code) tuples. Modules
        contribute proportionally to their (filtered) function count, longer
        functions are preferred within a module when PREFER_LONGER_FUNCTIONS
        is True, and dunder/trivial functions are skipped first.
        """
        rng = random.Random(SAMPLE_RANDOM_SEED)

        # Step 1: build candidate pool per module after filtering.
        per_module_candidates: Dict[str, List[Tuple[str, Dict, str, int]]] = {}
        for module, funcs in function_summaries.items():
            ast_funcs = ast_results.get(module, {}).get('functions', [])
            ast_index = {f.get('name'): f for f in ast_funcs}

            module_pool: List[Tuple[str, Dict, str, int]] = []
            for fn_name, summary in funcs.items():
                if SKIP_DUNDER_METHODS and fn_name.startswith("__") and fn_name.endswith("__"):
                    continue
                ast_fn = ast_index.get(fn_name) or {}
                fn_code = ast_fn.get('code') or ''
                line_count = ast_fn.get('line_count', 0) or len(fn_code.splitlines())
                if line_count < MIN_FUNCTION_LINES:
                    continue
                if not fn_code:
                    continue
                module_pool.append((fn_name, summary, fn_code, line_count))

            if module_pool:
                per_module_candidates[module] = module_pool

        if not per_module_candidates:
            return []

        # Step 2: allocate quota per module proportionally to pool size.
        total_candidates = sum(len(p) for p in per_module_candidates.values())
        if total_candidates <= target_size:
            quotas = {m: len(p) for m, p in per_module_candidates.items()}
        else:
            quotas = {}
            for m, pool in per_module_candidates.items():
                share = max(1, round(target_size * len(pool) / total_candidates))
                quotas[m] = min(share, len(pool))

            while sum(quotas.values()) > target_size:
                
                biggest = max(quotas, key=quotas.get)
                if quotas[biggest] <= 1:
                    break
                quotas[biggest] -= 1

        # Step 3: pick within each module.
        sampled: List[Tuple[str, str, Dict, str]] = []
        for module, pool in per_module_candidates.items():
            quota = quotas.get(module, 0)
            if quota <= 0:
                continue
            if PREFER_LONGER_FUNCTIONS:
                # Weight by line count so longer functions are likelier.
                weights = [lc for (_, _, _, lc) in pool]
                
                weights = [max(1, w) for w in weights]
                picks: List[int] = []
                indices = list(range(len(pool)))
               
                while len(picks) < quota and indices:
                    chosen_idx = rng.choices(indices, weights=[weights[i] for i in indices], k=1)[0]
                    picks.append(chosen_idx)
                    indices.remove(chosen_idx)
                chosen = [pool[i] for i in picks]
            else:
                chosen = rng.sample(pool, quota)

            for fn_name, summary, fn_code, _ in chosen:
                sampled.append((module, fn_name, summary, fn_code))

        return sampled

    # ---------------- Top-level orchestration ----------------

    def evaluate_repository(self, context_file: str) -> Dict:
        print(f"\n{'=' * 80}")
        print(f" Evaluating with judge model: {self.judge_model}")
        print(f"{'=' * 80}")

        with open(context_file) as f:
            context = json.load(f)

        repo_name = context.get('repo_name', 'Unknown')
        ast_results = context.get('ast_results', {})
        repo_summary = context.get('repo_summary', {})
        module_summaries = context.get('module_summaries', {})
        function_summaries = context.get('function_summaries', {})

        print(f"\n Repository: {repo_name}")

        results = {
            "repo_name": repo_name,
            "judge_model": self.judge_model,
            "generator_model": "Claude Haiku 4.5",
            "repo_summary_evaluation": None,
            "module_summaries_evaluation": None,
            "function_summaries": {},
            "architecture_evaluation": None,
            "overall_metrics": {},
        }

        if repo_summary:
            print(f"\n Step 1/4: Evaluating repository summary")
            results['repo_summary_evaluation'] = self.evaluate_repo_summary(
                ast_results, repo_summary, module_summaries
            )

        if module_summaries:
            print(f"\n Step 2/4: Evaluating module summaries")
            results['module_summaries_evaluation'] = self.evaluate_module_summaries(
                ast_results, module_summaries, function_summaries
            )

        if function_summaries:
            print(f"\n Step 3/4: Evaluating function summaries (stratified sample)")
            sampled = self._sample_functions(function_summaries, ast_results)
            total_funcs = sum(len(f) for f in function_summaries.values())
            print(f"   Sampled {len(sampled)} of {total_funcs} functions across "
                  f"{len({m for m, _, _, _ in sampled})} modules")
            results['function_sampling'] = {
                "target_size": TARGET_FUNCTION_SAMPLE_SIZE,
                "sampled_count": len(sampled),
                "total_functions": total_funcs,
                "sampled_items": [f"{m}::{n}" for m, n, _, _ in sampled],
                "skip_dunders": SKIP_DUNDER_METHODS,
                "min_function_lines": MIN_FUNCTION_LINES,
                "prefer_longer": PREFER_LONGER_FUNCTIONS,
                "random_seed": SAMPLE_RANDOM_SEED,
            }
            for module, fn_name, summary, fn_code in sampled:
                print(f"   {module}::{fn_name}")
                results['function_summaries'][f"{module}::{fn_name}"] = (
                    self.evaluate_function_summary(fn_code, summary)
                )

        mermaid_path = Path(f"diagrams/mermaid_{repo_name}.mmd")
        if mermaid_path.exists():
            print(f"\n Step 4/4: Evaluating architecture diagram")
            mermaid_code = mermaid_path.read_text()
            files = [k for k in ast_results.keys() if k != '__analysis_summary__']
            results['architecture_evaluation'] = self.evaluate_architecture_diagram(
                files, mermaid_code, context.get('logical_groups', {})
            )

        self._calculate_metrics(results)
        return results

    def _calculate_metrics(self, results: Dict):
        """Aggregate scores into a single overall_quality value."""
        m = {
            "repo_summary_score": 0,
            "module_summary_avg": 0,
            "function_summary_avg": 0,
            "architecture_score": 0,
            "overall_quality": 0,
        }

        if results['repo_summary_evaluation']:
            m['repo_summary_score'] = results['repo_summary_evaluation']['overall_score']
        if results['module_summaries_evaluation']:
            m['module_summary_avg'] = results['module_summaries_evaluation']['average_score']
        if results['function_summaries']:
            scores = [e['overall_score'] for e in results['function_summaries'].values()]
            m['function_summary_avg'] = round(sum(scores) / len(scores), 2) if scores else 0
        if results['architecture_evaluation']:
            m['architecture_score'] = results['architecture_evaluation']['overall_score']

        
        m['overall_quality'] = round(
            m['repo_summary_score'] * 0.20
            + m['module_summary_avg'] * 0.25
            + m['function_summary_avg'] * 0.35
            + m['architecture_score'] * 0.20,
            2,
        )
        results['overall_metrics'] = m


def resolve_context_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    candidate = Path(f"context_{arg}.json")
    if candidate.is_file():
        return candidate
    print(f"Error: could not find context file for '{arg}'")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    context_path = resolve_context_path(sys.argv[1])
    results = OpusEvaluator().evaluate_repository(str(context_path))

    output_file = str(context_path).replace("context_", "evaluation_")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    m = results['overall_metrics']
    print(f"\n{'=' * 80}")
    print(" EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n Overall Metrics (1-5):")
    print(f"   Repo Summary:        {m['repo_summary_score']:.2f}")
    print(f"   Module Summaries:    {m['module_summary_avg']:.2f}")
    print(f"   Function Summaries:  {m['function_summary_avg']:.2f}")
    print(f"   Architecture:        {m['architecture_score']:.2f}")
    print(f"\n   Overall Quality:     {m['overall_quality']:.2f}/5.0")
    print(f"\n Results saved to: {output_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
