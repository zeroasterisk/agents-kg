"""Canonical seed entities for the agentic web ecosystem.

These are well-known entities that the extraction prompt should match against
rather than creating new duplicates. The seed list serves two purposes:
1. Injected into the extraction prompt so the LLM prefers existing entity_ids
2. Used by entity resolution to merge fuzzy matches to canonical forms
"""

SEED_ENTITIES = [
    # --- Organizations ---
    {"entity_id": "organization:google", "name": "Google", "type": "Organization", "kind": "company",
     "aliases": ["Google LLC", "Google Cloud", "Google DeepMind", "Google AI"]},
    {"entity_id": "organization:anthropic", "name": "Anthropic", "type": "Organization", "kind": "company",
     "aliases": ["Anthropic PBC"]},
    {"entity_id": "organization:openai", "name": "OpenAI", "type": "Organization", "kind": "company",
     "aliases": ["Open AI"]},
    {"entity_id": "organization:microsoft", "name": "Microsoft", "type": "Organization", "kind": "company",
     "aliases": ["Microsoft Corp", "MSFT"]},
    {"entity_id": "organization:meta", "name": "Meta", "type": "Organization", "kind": "company",
     "aliases": ["Meta Platforms", "Facebook"]},
    {"entity_id": "organization:linux-foundation", "name": "Linux Foundation", "type": "Organization", "kind": "foundation",
     "aliases": ["LF", "Linux Foundation AI"]},
    {"entity_id": "organization:ietf", "name": "IETF", "type": "Organization", "kind": "standards_body",
     "aliases": ["Internet Engineering Task Force"]},
    {"entity_id": "organization:w3c", "name": "W3C", "type": "Organization", "kind": "standards_body",
     "aliases": ["World Wide Web Consortium"]},
    {"entity_id": "organization:aaif", "name": "AAIF", "type": "Organization", "kind": "foundation",
     "aliases": ["Agent Attestation & Interoperability Forum"]},
    {"entity_id": "organization:agntcy", "name": "AGNTCY", "type": "Organization", "kind": "consortium",
     "aliases": ["The Agency"]},
    {"entity_id": "organization:ibm", "name": "IBM", "type": "Organization", "kind": "company",
     "aliases": ["IBM Research"]},
    {"entity_id": "organization:amazon", "name": "Amazon", "type": "Organization", "kind": "company",
     "aliases": ["AWS", "Amazon Web Services"]},
    {"entity_id": "organization:apple", "name": "Apple", "type": "Organization", "kind": "company",
     "aliases": ["Apple Inc"]},
    {"entity_id": "organization:nvidia", "name": "NVIDIA", "type": "Organization", "kind": "company",
     "aliases": ["Nvidia"]},
    {"entity_id": "organization:langchain", "name": "LangChain", "type": "Organization", "kind": "company",
     "aliases": ["LangChain Inc"]},
    {"entity_id": "organization:modelcontextprotocol", "name": "MCP Organization", "type": "Organization", "kind": "consortium",
     "aliases": ["modelcontextprotocol GitHub org"]},
    {"entity_id": "organization:kaggle", "name": "Kaggle", "type": "Organization", "kind": "company",
     "aliases": []},

    # --- Groups ---
    {"entity_id": "group:agents-wg", "name": "Agents Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Agents WG"]},
    {"entity_id": "group:auth-wg", "name": "Auth Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Auth WG"]},
    {"entity_id": "group:enterprise-wg", "name": "Enterprise Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Enterprise WG"]},
    {"entity_id": "group:governance-wg", "name": "Governance Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Governance WG"]},
    {"entity_id": "group:server-card-wg", "name": "Server Card Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Server Card WG"]},
    {"entity_id": "group:server-identity-wg", "name": "Server Identity Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Server Identity WG"]},
    {"entity_id": "group:transports-wg", "name": "Transports Working Group", "type": "Group", "kind": "wg",
     "aliases": ["Transports WG"]},

    # --- Protocols ---
    {"entity_id": "protocol:mcp", "name": "Model Context Protocol", "type": "Protocol", "kind": "spec",
     "aliases": ["MCP"]},
    {"entity_id": "protocol:a2a", "name": "Agent-to-Agent Protocol", "type": "Protocol", "kind": "spec",
     "aliases": ["A2A", "Agent2Agent", "agent-to-agent"]},
    {"entity_id": "protocol:a2ui", "name": "A2UI", "type": "Protocol", "kind": "spec",
     "aliases": ["Agent-to-UI", "GenUI"]},
    {"entity_id": "protocol:ag-ui", "name": "AG-UI", "type": "Protocol", "kind": "spec",
     "aliases": ["Agent-UI Protocol", "CopilotKit AG-UI"]},
    {"entity_id": "protocol:ibm-acp", "name": "IBM Agent Communication Protocol", "type": "Protocol", "kind": "spec",
     "description": "IBM's ACP which was aligned/de-duplicated with A2A",
     "aliases": ["IBM ACP"]},
    {"entity_id": "protocol:openai-acp", "name": "OpenAI Agent Payments Protocol", "type": "Protocol", "kind": "spec",
     "description": "OpenAI's payments-focused ACP",
     "aliases": ["OpenAI ACP", "Payments ACP"]},
    {"entity_id": "protocol:zed-acp", "name": "Zed Agent Context Protocol", "type": "Protocol", "kind": "spec",
     "description": "Zed's local stdio-based Agent Context Protocol",
     "aliases": ["Zed ACP", "Agent Context Protocol"]},
    {"entity_id": "protocol:openapi", "name": "OpenAPI Specification", "type": "Protocol", "kind": "standard",
     "aliases": ["OpenAPI", "OAS", "Swagger"]},
    {"entity_id": "protocol:json-rpc-2.0", "name": "JSON-RPC 2.0", "type": "Protocol", "kind": "standard",
     "aliases": ["JSON-RPC"]},
    {"entity_id": "protocol:oauth-2.1", "name": "OAuth 2.1", "type": "Protocol", "kind": "standard",
     "aliases": ["OAuth", "OAuth 2.0"]},
    {"entity_id": "protocol:openid-connect", "name": "OpenID Connect", "type": "Protocol", "kind": "standard",
     "aliases": ["OIDC"]},
    {"entity_id": "protocol:spiffe", "name": "SPIFFE", "type": "Protocol", "kind": "standard",
     "aliases": ["Secure Production Identity Framework for Everyone"]},
    {"entity_id": "protocol:http-402", "name": "HTTP 402", "type": "Protocol", "kind": "standard",
     "aliases": ["Payment Required"]},
    {"entity_id": "protocol:agent-payments", "name": "Agent Payments Protocol", "type": "Protocol", "kind": "spec",
     "aliases": ["x402"]},

    # --- Projects ---
    {"entity_id": "project:adk", "name": "Agent Development Kit", "type": "Project", "kind": "framework",
     "aliases": ["ADK", "google/adk-python", "google/adk-docs"]},
    {"entity_id": "project:adk-elixir", "name": "ADK Elixir", "type": "Project", "kind": "framework",
     "aliases": ["adk-elixir", "ADK for Elixir"],
     "description": "OTP-native AI agent framework inspired by Google ADK, built for the BEAM."},

    {"entity_id": "project:gemini", "name": "Gemini", "type": "Project", "kind": "platform",
     "aliases": ["Gemini 2.0", "Gemini 2.5", "Gemini Pro", "Gemini Flash"]},
    {"entity_id": "project:gemini-2.5-pro", "name": "Gemini 2.5 Pro", "type": "Project", "kind": "platform",
     "aliases": ["Gemini 2.5 Pro"]},
    {"entity_id": "project:gemini-2.5-flash", "name": "Gemini 2.5 Flash", "type": "Project", "kind": "platform",
     "aliases": ["Gemini 2.5 Flash"]},
    {"entity_id": "project:vertex-ai", "name": "Vertex AI", "type": "Project", "kind": "platform",
     "aliases": ["Vertex AI Agent Engine", "Vertex AI Agent Builder"]},
    {"entity_id": "project:claude", "name": "Claude", "type": "Project", "kind": "platform",
     "aliases": ["Claude 3", "Claude Code", "Claude Desktop"]},
    {"entity_id": "project:chatgpt", "name": "ChatGPT", "type": "Project", "kind": "platform",
     "aliases": ["GPT-4", "GPT-4o"]},
    {"entity_id": "project:langchain", "name": "LangChain", "type": "Project", "kind": "framework",
     "aliases": ["LangSmith"]},
    {"entity_id": "project:langgraph", "name": "LangGraph", "type": "Project", "kind": "framework",
     "aliases": []},
    {"entity_id": "project:mcp-sdk-typescript", "name": "MCP TypeScript SDK", "type": "Project", "kind": "sdk",
     "aliases": ["@modelcontextprotocol/sdk", "MCP TS SDK"]},
    {"entity_id": "project:mcp-sdk-python", "name": "MCP Python SDK", "type": "Project", "kind": "sdk",
     "aliases": ["mcp Python SDK"]},
    {"entity_id": "project:mcp-inspector", "name": "MCP Inspector", "type": "Project", "kind": "tool",
     "aliases": ["Inspector"]},
    {"entity_id": "project:opentelemetry", "name": "OpenTelemetry", "type": "Project", "kind": "framework",
     "aliases": ["OTel", "OTEL"]},
    {"entity_id": "project:beeai", "name": "BeeAI", "type": "Project", "kind": "framework",
     "aliases": ["Bee Agent Framework"]},
    {"entity_id": "project:cloud-run", "name": "Cloud Run", "type": "Project", "kind": "platform",
     "aliases": ["Google Cloud Run"]},
    {"entity_id": "project:react", "name": "ReAct", "type": "Project", "kind": "framework",
     "aliases": ["ReAct prompting", "Reasoning + Acting"]},
    {"entity_id": "project:chain-of-thought", "name": "Chain-of-Thought", "type": "Project", "kind": "framework",
     "aliases": ["CoT", "CoT prompting"]},
    {"entity_id": "project:rag", "name": "Retrieval-Augmented Generation", "type": "Project", "kind": "framework",
     "aliases": ["RAG"]},
    {"entity_id": "project:model-armor", "name": "Model Armor", "type": "Project", "kind": "tool",
     "aliases": []},
    {"entity_id": "project:gemma", "name": "Gemma", "type": "Project", "kind": "library",
     "aliases": ["Gemma 2"]},

    # --- Capabilities ---
    {"entity_id": "capability:tool-use", "name": "Tool Use", "type": "Capability",
     "aliases": ["function calling", "tool calling"]},
    {"entity_id": "capability:reasoning", "name": "Reasoning", "type": "Capability",
     "aliases": ["chain-of-thought reasoning"]},
    {"entity_id": "capability:planning", "name": "Planning", "type": "Capability",
     "aliases": ["task planning", "multi-step planning"]},
    {"entity_id": "capability:memory", "name": "Memory", "type": "Capability",
     "aliases": ["context memory", "long-term memory", "short-term memory"]},
    {"entity_id": "capability:multi-agent", "name": "Multi-Agent Systems", "type": "Capability",
     "aliases": ["multi-agent", "MAS", "multi-agent collaboration"]},
    {"entity_id": "capability:observability", "name": "Observability", "type": "Capability",
     "aliases": ["monitoring", "tracing", "logging"]},
    {"entity_id": "capability:security", "name": "Security", "type": "Capability",
     "aliases": ["agent security", "safety"]},
    {"entity_id": "capability:authentication", "name": "Authentication", "type": "Capability",
     "aliases": ["auth", "identity verification"]},
    {"entity_id": "capability:authorization", "name": "Authorization", "type": "Capability",
     "aliases": ["access control", "permissions"]},
    {"entity_id": "capability:human-in-the-loop", "name": "Human-in-the-Loop", "type": "Capability",
     "aliases": ["HITL", "human oversight", "human review"]},

    # --- People ---
    {"entity_id": "person:alan-blount", "name": "Alan Blount", "type": "Person",
     "aliases": ["@zeroasterisk"]},
    {"entity_id": "person:julia-wiesinger", "name": "Julia Wiesinger", "type": "Person",
     "aliases": []},
    {"entity_id": "person:patrick-marlow", "name": "Patrick Marlow", "type": "Person",
     "aliases": []},
    {"entity_id": "person:antonio-gulli", "name": "Antonio Gulli", "type": "Person",
     "aliases": []},
    {"entity_id": "person:shubham-saboo", "name": "Shubham Saboo", "type": "Person",
     "aliases": []},
]


def get_seed_entities() -> list[dict]:
    """Return the canonical seed entities."""
    return SEED_ENTITIES


def seed_entity_ids() -> set[str]:
    """Return set of all canonical entity_ids for quick lookup."""
    return {e["entity_id"] for e in SEED_ENTITIES}


def format_seed_for_prompt() -> str:
    """Format seed entities as a compact reference for the extraction prompt."""
    lines = []
    by_type = {}
    for e in SEED_ENTITIES:
        by_type.setdefault(e["type"], []).append(e)

    for etype in ["Organization", "Protocol", "Project", "Capability", "Person"]:
        entries = by_type.get(etype, [])
        if not entries:
            continue
        lines.append(f"\n{etype}:")
        for e in entries:
            aliases = ", ".join(e.get("aliases", []))
            alias_str = f" (aka: {aliases})" if aliases else ""
            kind = f"/{e['kind']}" if e.get("kind") else ""
            lines.append(f"  {e['entity_id']}{kind} — {e['name']}{alias_str}")

    return "\n".join(lines)
