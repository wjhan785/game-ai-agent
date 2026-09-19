"""Greedy-LLM baseline: the full agent with planner and ledger readback off.
Everything else is identical, so the gap measures what planning adds."""

from agent.outer_loop import SessionConfig

METHOD = "greedy_llm"


def greedy_llm_config(run_name: str, **kwargs) -> SessionConfig:
    return SessionConfig(run_name=run_name, planner=False, method=METHOD, **kwargs)
