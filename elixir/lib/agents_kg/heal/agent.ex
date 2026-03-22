defmodule AgentsKg.Heal.Agent do
  @moduledoc """
  Healer agent for Agents-KG extractions.
  Uses a more capable model to fix extractions that failed QA.
  """
  
  alias ADK.Agent.LlmAgent
  alias ADK.Runner
  alias ADK.Event

  @system_prompt """
  You are an expert knowledge graph extraction repair system.
  Your job is to fix a flawed extraction of entities and relationships from a text chunk.
  
  You will receive:
  1. The ORIGINAL TEXT.
  2. The FLAWED EXTRACTION (JSON).
  3. The QA REVIEW detailing what needs to be fixed.
  
  ## REPAIR INSTRUCTIONS:
  - Address all issues listed in the QA REVIEW.
  - ONLY these Node Types: Organization, Group, Person, Project, Protocol, Capability.
  - ONLY these Edge Types: MEMBER_OF, GOVERNS, DEVELOPS, IMPLEMENTS, COMPETES_WITH, ADDRESSES, AUTHORED, CHAIRS, SPONSORS, PART_OF, SUPERSEDES, CONTRIBUTES_TO, DEFINES, COMPLEMENTS, USES.
  - entity_id MUST be "type:kebab-case-name" without a kind suffix.
  
  Output the FIXED extraction as valid JSON matching this schema:
  {
    "entities": [
      {
        "entity_id": "type:kebab-case-name",
        "name": "Display Name",
        "type": "Organization|Group|Person|Project|Protocol|Capability",
        "kind": "specific kind or null",
        "description": "One sentence description",
        "aliases": ["alt name 1"]
      }
    ],
    "edges": [
      {
        "source_entity_id": "type:name",
        "target_entity_id": "type:name",
        "edge_type": "DEVELOPS",
        "confidence": 0.9,
        "properties": {}
      }
    ]
  }
  """

  def new(opts \\ []) do
    base_opts = [
      name: "HealAgent",
      # Using a more capable, expensive model for self-healing escalation
      model: "gemini-2.5-pro",
      instruction: @system_prompt,
      generate_config: %{
        response_mime_type: "application/json",
        temperature: 0.2
      }
    ]

    LlmAgent.new(Keyword.merge(base_opts, opts))
  end

  def run(text, flawed_extraction, qa_review, session_id \\ nil, agent_opts \\ []) do
    agent = new(agent_opts)
    runner = Runner.new(app_name: "heal", agent: agent)
    session_id = session_id || "heal_#{System.unique_integer([:positive])}"
    
    message = """
    ORIGINAL TEXT:
    #{text}
    
    FLAWED EXTRACTION:
    #{Jason.encode!(flawed_extraction, pretty: true)}
    
    QA REVIEW (FIX THESE):
    #{Jason.encode!(qa_review, pretty: true)}
    """

    events = Runner.run(runner, "system", session_id, message)

    text_response =
      events
      |> Enum.filter(&(&1.author == "HealAgent"))
      |> Enum.map(&Event.text/1)
      |> Enum.reject(&is_nil/1)
      |> Enum.join("\n")

    case Jason.decode(text_response) do
      {:ok, parsed} ->
        {:ok, parsed}
      {:error, reason} ->
        {:error, {:json_decode_error, reason, text_response}}
    end
  end
end
