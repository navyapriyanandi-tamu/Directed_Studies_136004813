#!/usr/bin/env python3
"""
CodeChat - Streamlit Dashboard
Interactive web interface for repository analysis with chat and visualizations

Usage: streamlit run streamlit_app.py
"""

import sys
import json
from pathlib import Path
import streamlit as st
import config

# Page configuration
st.set_page_config(
    page_title="CodeChat - Code Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_context_files():
    """Load all available context files"""
    contexts = {}
    for ctx_file in Path(".").glob("context_*.json"):
        repo_name = ctx_file.stem.replace("context_", "")
        contexts[repo_name] = str(ctx_file)
    return contexts


def load_context(context_file):
    """Load context data from JSON file"""
    with open(context_file, 'r') as f:
        return json.load(f)


def display_overview(context):
    """Display repository overview"""
    st.header("Repository Overview")

    repo_summary = context.get("repo_summary", {})
    summary = context.get("ast_results", {}).get("__analysis_summary__", {})

    # Stats row
    col1, col2, col3 = st.columns(3)
    col1.metric("Files", summary.get("total_modules", 0))
    col2.metric("Functions", summary.get("total_functions", 0))
    col3.metric("Classes", summary.get("total_classes", 0))

    # Repository Summary
    st.markdown("---")
    st.subheader("Repository Summary")

    if repo_summary and 'human' in repo_summary:
        st.write(repo_summary['human'])
    else:
        st.info("No summary available")

    if repo_summary and 'technical' in repo_summary:
        with st.expander("Technical Summary"):
            st.write(repo_summary['technical'])


def display_modules(context):
    """Display all modules with summaries"""
    st.header("Modules")

    module_summaries = context.get("module_summaries", {})

    if not module_summaries:
        st.warning("No module summaries found")
        return

    search = st.text_input("Search modules", placeholder="e.g., auth, storage, validate")

    for module_name, summary in sorted(module_summaries.items()):
        if search and search.lower() not in module_name.lower():
            continue

        with st.expander(module_name):
            if 'human' in summary:
                st.write(summary['human'])
            if 'technical' in summary:
                with st.expander("Technical Details"):
                    st.write(summary['technical'])


def display_functions(context):
    """Display all functions with summaries"""
    st.header("Functions")

    function_summaries = context.get("function_summaries", {})

    if not function_summaries:
        st.warning("No function summaries found")
        return

    search = st.text_input("Search functions", placeholder="e.g., validate, save, process")

    total_functions = sum(len(funcs) for funcs in function_summaries.values())
    st.caption(f"Total functions: {total_functions}")

    for module_name, functions in sorted(function_summaries.items()):
        if search:
            matching_funcs = {name: s for name, s in functions.items()
                            if search.lower() in name.lower()}
            if not matching_funcs:
                continue
            functions = matching_funcs

        with st.expander(f"{module_name} ({len(functions)} functions)"):
            for func_name, summary in sorted(functions.items()):
                st.markdown(f"### `{func_name}()`")

                if isinstance(summary, dict):
                    if 'human' in summary:
                        st.write(summary['human'])
                    if 'technical' in summary:
                        with st.expander("Technical Details"):
                            st.write(summary['technical'])
                else:
                    st.write(summary)

                st.markdown("---")


def display_architecture(context):
    """Display architecture diagram"""
    st.header("Architecture Diagram")

    repo_name = context.get("repo_name", "Unknown")
    mermaid_file = Path(f"diagrams/mermaid_{repo_name}.mmd")

    if mermaid_file.exists():
        with open(mermaid_file, 'r') as f:
            mermaid_code = f.read()

        # Create mermaid.live link with pre-loaded diagram
        import base64

        state = {
            "code": mermaid_code,
            "mermaid": {"theme": "default"},
            "autoSync": True,
            "updateDiagram": True
        }
        state_json = json.dumps(state)
        encoded_state = base64.b64encode(state_json.encode('utf-8')).decode('utf-8')
        mermaid_link = f"https://mermaid.live/edit#base64:{encoded_state}"

        st.subheader("Visual Architecture")
        st.markdown(f"**[Click here to view interactive diagram in Mermaid Live]({mermaid_link})**")
        st.markdown("---")

        st.code(mermaid_code, language="mermaid")

        # Logical groups
        logical_groups = context.get("logical_groups", {})
        if logical_groups:
            st.markdown("---")
            st.subheader("Logical Groups")
            for group_name, files in logical_groups.items():
                with st.expander(f"{group_name} ({len(files)} files)"):
                    for file in sorted(files):
                        st.write(f"- {file}")
    else:
        st.warning("No architecture diagram found")
        st.info(f"Run: `python3 analyze_any_repo.py <repo_path>` to generate diagrams")


def display_dependencies(context):
    """Display module dependencies"""
    st.header("Module Dependencies")

    repo_name = context.get("repo_name", "Unknown")
    deps_file = Path(f"diagrams/dependencies_{repo_name}.txt")

    if deps_file.exists():
        with open(deps_file, 'r') as f:
            deps_content = f.read()

        st.code(deps_content, language="text")
    else:
        st.warning("No dependency graph found")
        st.info(f"Run: `python3 analyze_any_repo.py <repo_path>` to generate diagrams")


def chat_interface(context):
    """Chat interface for asking questions"""
    st.header("Ask Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about this codebase"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = handle_query(prompt, context)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def build_context_summary(context: dict) -> str:
    """Build a concise text summary of the repo context for the LLM"""
    repo_name = context.get("repo_name", "Unknown")
    repo_summary = context.get("repo_summary", {})
    module_summaries = context.get("module_summaries", {})
    function_summaries = context.get("function_summaries", {})
    ast_summary = context.get("ast_results", {}).get("__analysis_summary__", {})

    parts = []
    parts.append(f"Repository: {repo_name}")
    parts.append(f"Stats: {ast_summary.get('total_modules', 0)} modules, {ast_summary.get('total_functions', 0)} functions, {ast_summary.get('total_classes', 0)} classes")

    if repo_summary:
        parts.append(f"\nRepo Summary: {repo_summary.get('human', '')}")
        parts.append(f"Technical: {repo_summary.get('technical', '')}")

    if module_summaries:
        parts.append(f"\nModules:")
        for mod_path, summary in module_summaries.items():
            parts.append(f"- {mod_path}: {summary.get('human', '')}")

    if function_summaries:
        parts.append(f"\nFunctions:")
        for mod_path, funcs in function_summaries.items():
            for func_name, summary in funcs.items():
                desc = summary.get('human', '') if isinstance(summary, dict) else str(summary)
                parts.append(f"- {mod_path}::{func_name}(): {desc}")

    return "\n".join(parts)


def handle_query(query: str, context: dict) -> str:
    """Answer user question directly using LLM with full repo context"""
    if not config.TAMUS_AI_CHAT_API_KEY:
        return "API key not configured. Set TAMUS_AI_CHAT_API_KEY environment variable."

    try:
        client = config.get_client()
        context_text = build_context_summary(context)

        prompt = f"""
You are a knowledgeable assistant helping a developer understand a Python codebase. You have access to a structured analysis of the repository — module-level and function-level   
summaries, AST statistics, and logical groupings. Use this analysis as the source of truth for your answer, do not invent details about the codebase that aren't in the analysis.

If the analysis doesn't contain enough information to answer confidently, say so explicitly (e.g., "The summary doesn't cover this directly, but based on…") rather than guessing. Cite specific module 
or function names from the analysis when they support your answer.

Be direct: answer the question first, then add context only if it genuinely helps. Avoid restating the question, padding with disclaimers, or summarizing the whole repository when the question is     
narrow. Format your response using markdown.

--- REPOSITORY ANALYSIS ---
{context_text}
--- END ---

User question: {query}"""

        return config.llm_call(
            client, config.LLM_MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=2000, temperature=1
        )

    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main Streamlit app"""

    default_repo = sys.argv[1] if len(sys.argv) > 1 else None

    with st.sidebar:
        st.title("CodeChat")
        st.markdown("---")

        contexts = load_context_files()

        if not contexts:
            st.error("No context files found!")
            st.info("Run: `python3 codechat.py <repo_path>`")
            st.stop()

        repo_list = list(contexts.keys())
        default_index = 0
        if default_repo and default_repo in repo_list:
            default_index = repo_list.index(default_repo)

        selected_repo = st.selectbox(
            "Select Repository",
            options=repo_list,
            index=default_index,
            key="repo_selector"
        )

        context_file = contexts[selected_repo]
        context = load_context(context_file)

        st.markdown("---")

        page = st.radio(
            "Go to:",
            options=["Chat", "Overview", "Architecture", "Dependencies", "Functions", "Modules"],
            key="navigation"
        )

    if page == "Chat":
        chat_interface(context)
    elif page == "Overview":
        display_overview(context)
    elif page == "Architecture":
        display_architecture(context)
    elif page == "Dependencies":
        display_dependencies(context)
    elif page == "Functions":
        display_functions(context)
    elif page == "Modules":
        display_modules(context)


if __name__ == "__main__":
    main()
