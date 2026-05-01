"""
Module Grouper - Groups modules by logical purpose using LLM
Analyzes module summaries to create architectural groupings
"""

import json
from typing import Dict, List
import config


class ModuleGrouper:
    """
    Groups modules by logical architectural purpose using LLM analysis

    """

    def __init__(self):
        """Initialize with API configuration from config.py"""
        self.client = config.get_client()
        self.model = config.LLM_MODEL

    def group_modules(self, module_summaries: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Analyze module summaries and group by logical purpose
        """

        if not module_summaries:
            print("   No module summaries available")
            return {"Ungrouped": []}

        # Build prompt with module summaries
        prompt = self._build_grouping_prompt(module_summaries)

        # Get LLM response
        try:
            print(f"    Asking LLM to group {len(module_summaries)} modules")
            # Parse JSON response
            response_text = config.llm_call(self.client, self.model, [{"role": "user", "content": prompt}], max_tokens=2000, temperature=1)
            groups = self._parse_response(response_text, module_summaries)

            print(f"    LLM created {len(groups)} logical groups")
            return groups

        except Exception as e:
            print(f"    LLM grouping failed: {e}")
            # Fallback: group by directory structure
            return self._fallback_grouping(module_summaries)

    def _build_grouping_prompt(self, module_summaries: Dict) -> str:
        """Build prompt for LLM to group modules"""

        prompt = f"""You are analyzing a Python codebase to create logical architectural groups.

I have {len(module_summaries)} modules. Your task is to group them by their LOGICAL PURPOSE (not directory structure).

**Modules:**

"""
        # Add each module with its human summary
        for module_path, summary in module_summaries.items():
            human_summary = summary.get('human', 'No summary available')
            prompt += f"- `{module_path}`: {human_summary}\n"

        prompt += f"""

**Your Task:**

Group these modules into logical architectural categories. Common categories include:
- "Application Entry" (main files, CLI entry points, application startup)
- "Authentication & Security" (login, permissions, auth, security)
- "Business Logic" (core domain logic, operations, handlers)
- "Data Storage" (database, file storage, persistence, repositories)
- "API Layer" (routes, endpoints, controllers, handlers)
- "User Interface" (UI components, views, templates)
- "Utilities" (helpers, validators, formatters, common tools)
- "Configuration" (settings, config files, environment)
- "Testing" (test files, test utilities)
- "Models" (data models, entities, schemas)

The list above is just inspiration. If a category from the list doesn't fit this codebase, omit it. 
If a different name fits better (e.g., "Transport Layer", "Language Processing Core", "Request Pipeline"), invent it. 
Choose names that reflect what THIS repository actually does.

**Guidelines:**
1. Create an appropriate number of logical groups for this {len(module_summaries)}-module repository
2. Balance clarity (not too many groups) with accuracy (not too few)
3. Aim for groups that represent major architectural components
4. Typically 3-10 groups works well, but use your judgment
5. Each module should be in exactly ONE group
6. Group by PURPOSE and FUNCTIONALITY, not by directory structure

**Output Format:**

Return a JSON object where keys are category names and values are arrays of module paths:

```json
{{
  "Application Entry": ["main.py", "cli.py"],
  "Business Logic": ["handlers/task_handler.py"],
  "Data Storage": ["storage/database.py", "storage/file_storage.py"],
  "Utilities": ["utils/logger.py", "utils/validators.py"]
}}
```

**Important:**
- Use the EXACT module paths from the list above
- Return ONLY the JSON object, no prose, no markdown explanation, no commentary before or after
- Make sure all modules are included in exactly one group, and no module appears in multiple groups
"""
        return prompt

    def _parse_response(self, response_text: str, module_summaries: Dict) -> Dict[str, List[str]]:
        """Extract JSON from LLM response and validate"""
        try:
            # Handle markdown code blocks
            json_text = response_text.strip()
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()

            groups = json.loads(json_text)

            # Validate that all modules are included
            all_modules = set(module_summaries.keys())
            grouped_modules = set()
            for modules in groups.values():
                grouped_modules.update(modules)

            # Add any missing modules to "Other" category
            missing = all_modules - grouped_modules
            if missing:
                if "Other" not in groups:
                    groups["Other"] = []
                groups["Other"].extend(list(missing))
                print(f"    Added {len(missing)} ungrouped modules to 'Other' category")

            return groups

        except Exception as e:
            print(f"    Error parsing LLM response: {e}")
            print(f"    Response was: {response_text[:200]}")
            return self._fallback_grouping(module_summaries)

    def _fallback_grouping(self, module_summaries: Dict) -> Dict[str, List[str]]:
        """Fallback: Group by directory structure if LLM fails"""
        print("    Using fallback: grouping by directory structure")
        groups = {}

        for module_path in module_summaries.keys():
            if '/' in module_path:
                # Get first directory and format as title
                category = module_path.split('/')[0].replace('_', ' ').title()
            else:
                # Root level files
                category = "Root Modules"

            if category not in groups:
                groups[category] = []
            groups[category].append(module_path)

        return groups
