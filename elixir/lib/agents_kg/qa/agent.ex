defmodule AgentsKg.Qa.Agent do
  @moduledoc """
  QA agent for Agents-KG extractions.
  Reviews extracted entities and edges against the ontology rules.
  """
  
  alias ADK.Agent.LlmAgent
  alias ADK.Runner
  alias ADK.Event

  @system_prompt """
  You are a strict QA critic for a knowledge graph extraction system.
  Your job is to review an extracted JSON (entities and edges) and the original text chunk.
  
  Evaluate if the extraction violates any of these strict rules:
  1. ONLY these Node Types are allowed: Organization, Group, Person, Project, Protocol, Capability.
  2. ONLY these Edge Types are allowed: MEMBER_OF, GOVERNS, DEVELOPS, IMPLEMENTS, COMPETES_WITH, ADDRESSES, AUTHORED, CHAIRS, SPONSORS, PART_OF, SUPERSEDES, CONTRIBUTES_TO, DEFINES, COMPLEMENTS, USES.
  3. entity_id MUST be "type:kebab-case-name" without a kind suffix (e.g., "project:mcp-sdk", NOT "project:mcp-sdk/library").
  4. Person entities cannot be generic roles (e.g., "domain expert").
  5. Edges must connect valid entity_ids that exist in the entities list or are well-known.
  
  Output ONLY valid JSON in this format:
  {
    "passed": boolean,
    "reason": "string explaining why it passed or failed",
    "fixes_needed": ["list of specific issues to fix, if failed"]
  }
  """

  def new(opts \\ []) do
    base_opts = [
      name: "QaAgent",
      model: "gemini-2.5-flash",
      instruction: @system_prompt,
      generate_config: %{
        response_mime_type: "application/json",
        temperature: 0.1
      }
    ]

    LlmAgent.new(Keyword.merge(base_opts, opts))
  end

  def run(text, extracted_json, session_id \\ nil, agent_opts \\ []) do
    agent = new(agent_opts)
    runner = Runner.new(app_name: "qa", agent: agent)
    session_id = session_id || "qa_#{System.unique_integer([:positive])}"
    
    message = """
    ORIGINAL TEXT:
    #{text}
    
    EXTRACTED JSON:
    #{Jason.encode!(extracted_json, pretty: true)}
    """

    events = Runner.run(runner, "system", session_id, message)

    text_response =
      events
      |> Enum.filter(&(&1.author == "QaAgent"))
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
