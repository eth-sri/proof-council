"""proofstack — modular agent layer for mathagents.

The package is intentionally async-first
and Pydantic-typed; every Agent subclass declares Inputs/Outputs models
and an async run() method.
"""
from proofstack.agent import Agent
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext
from proofstack.events import EventEmitter, JSONLSink
from proofstack.state import ArtifactRef, ArtifactRegistry

__all__ = [
    "Agent",
    "ArtifactRef",
    "ArtifactRegistry",
    "BudgetExhausted",
    "BudgetSpec",
    "EventEmitter",
    "JSONLSink",
    "RunContext",
]
