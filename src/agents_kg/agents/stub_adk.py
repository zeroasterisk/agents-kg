"""Stub implementation of ADK Python agent for extraction.

This file provides a stub for an ADK agent to be used in the extraction stage,
allowing us to integrate ADK into the pipeline concept before full installation.
"""

class StubADKAgent:
    def __init__(self, model: str, instruction: str, tools: list = None):
        self.model = model
        self.instruction = instruction
        self.tools = tools or []

    def chat(self, message: str) -> str:
        """Mock chat method.
        
        In a real ADK agent, this would invoke the LLM and handle tool calls.
        Here we just simulate the call by returning a dummy JSON structure
        to not break the pipeline if called.
        """
        # In a real implementation, this would call the Gemini API or use ADK's internal runner
        return '{"entities": [], "edges": []}'

    def run(self, message: str):
        """Mock run method returning a stream of events."""
        yield {"event": "agent_started", "agent": "ExtractorAgent"}
        yield {"event": "llm_call", "model": self.model}
        yield {"event": "agent_finished", "output": self.chat(message)}
