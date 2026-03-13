"""Prompt builders for different answer styles."""

from __future__ import annotations


def _format_context(context: str) -> str:
    return context.strip() if context and context.strip() else "No external context available."

def build_concise_prompt(query: str, context: str) -> str:
    """Build a short-answer prompt."""
    return f""" You are a precise AI assistant. Use the provided context when relevant.
    Context: {_format_context(context)}
    User question: {query}
    Instructions: 
               - Give a short, direct answer.
               - Use 2-5 sentences unless a list is necessary.
               - If context is uncertain, state assumptions briefly. """.strip()

def build_detailed_prompt(query: str, context: str) -> str:
    """Build a structured long-form prompt."""

    return f""" You are an expert AI assistant. Use the provided context when relevant.

    Context: {_format_context(context)}
    User question: {query}
    Instructions:
               - Provide a detailed answer with clear section headers when useful.
               - Explain reasoning and important tradeoffs.
               - Use bullet points for clarity where appropriate.
               - If context is incomplete, explicitly call out uncertainty.""".strip()
