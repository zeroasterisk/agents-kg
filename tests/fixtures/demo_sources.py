"""Demo sources for testing and validation.

Each source includes expected entity types/counts for validation.
"""

DEMO_SOURCES = [
    {
        "url": "https://github.com/google/A2A/blob/main/README.md",
        "description": "A2A protocol README - simple page with clear entities",
        "expected_entity_types": ["Organization", "Protocol", "Project"],
        "expected_min_entities": 3,
        "expected_edge_types": ["DEVELOPS", "IMPLEMENTS"],
    },
    {
        "url": "https://google.github.io/A2A/latest/",
        "description": "A2A spec overview - protocol details, capabilities, architecture",
        "expected_entity_types": ["Organization", "Protocol", "Capability", "Project"],
        "expected_min_entities": 5,
        "expected_edge_types": ["DEVELOPS", "DEFINES", "ADDRESSES"],
    },
    {
        "url": "https://www.linuxfoundation.org/press/linux-foundation-launches-open-source-agentic-ai-initiative",
        "description": "LF AI announcement - multiple orgs, projects, protocols",
        "expected_entity_types": ["Organization", "Project", "Person", "Protocol"],
        "expected_min_entities": 5,
        "expected_edge_types": ["MEMBER_OF", "DEVELOPS", "SPONSORS"],
    },
]

# Simplified test source for quick validation
SIMPLE_TEST_SOURCE = {
    "url": "https://raw.githubusercontent.com/google/A2A/main/README.md",
    "description": "Raw README markdown - fastest to fetch and parse",
    "expected_entity_types": ["Organization", "Protocol"],
    "expected_min_entities": 2,
}
