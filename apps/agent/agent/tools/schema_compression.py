# Strips verbose Args/Returns/Use-this-when sections from tool descriptions
# before the LLM sees them, reducing the 3,161-token tool-schema footprint.
"""
Tool schema compression for Fortio.

Removes boilerplate sections from tool docstrings (Args, Returns,
Use this when, Examples) so only the essential 1–2 sentence description
reaches the LLM. This saves approximately 40 % of the per-call
tool-schema token budget without affecting tool execution — the full
original docstrings are still used at runtime by the ToolNode.

Usage
-----
    from agent.tools.schema_compression import compress_tool_schemas
    compressed_tools = compress_tool_schemas(ALL_TOOLS)
    llm.bind_tools(compressed_tools)
"""

from __future__ import annotations

import re

from langchain_core.tools import BaseTool

# Strip "Args:", "Returns:", "Use this when:", "Example(s):" and everything after.
_STRIP_SECTIONS_RE = re.compile(
    r"\n\s*(Args|Returns|Use this when|Examples?)\s*:\s*\n.*",
    re.DOTALL | re.IGNORECASE,
)


def _compress_description(description: str) -> str:
    """
    Remove boilerplate sections from a tool docstring, keeping the core
    description AND any "Use when" / trigger-example paragraph.

    The heuristic is:
      1. Strip everything from the first "Args:", "Returns:", "Use this when:",
         or "Examples:" heading onwards (case-insensitive).
      2. Keep the first TWO blank-line-separated paragraphs so that
         "Use when users ask:" sections (which teach the LLM *when* to call
         this tool) are preserved alongside the core description.
         Keeping only the first paragraph was too aggressive — it silently
         dropped trigger examples like 'How is my portfolio positioned if
         rates keep rising?' and caused the LLM to answer from training data
         instead of calling the correct tool.
      3. Return the stripped, trimmed result.
    """
    # Strip Args/Returns/Use-this-when and everything that follows
    stripped = _STRIP_SECTIONS_RE.sub("", description).strip()
    # Keep the first two paragraphs: core description + "Use when" examples.
    paragraphs = stripped.split("\n\n")
    return "\n\n".join(paragraphs[:2]).strip()


def compress_tool_schemas(tools: list[BaseTool]) -> list[BaseTool]:
    """
    Return a list of tools identical to the input except that each tool's
    .description has been compressed to its essential first paragraph.

    The original tool objects are NOT mutated — model_copy() creates a
    shallow copy with only the description field replaced.

    Args:
        tools: List of LangChain BaseTool instances (e.g. ALL_TOOLS).

    Returns:
        New list of tools with compressed descriptions. Safe to pass to
        llm.bind_tools() — the originals are still used by ToolNode.
    """
    compressed = []
    for t in tools:
        short_desc = _compress_description(t.description)
        compressed.append(t.model_copy(update={"description": short_desc}))
    return compressed
