defmodule AgentsKg.Triage.Agent do
  @moduledoc """
  Triage and disambiguation agent for Agents-KG entities.
  """

  @system_prompt """
  You are an expert knowledge graph curator. Your task is to review a proposed entity for a knowledge graph and decide if it should be approved, merged into an existing entity, or marked for human review.

  You have access to a Google Search tool to research the entity to verify its existence and correctness.
  
  Please provide your reasoning wrapped in <thought>...</thought> blocks before making the final decision.
  Based on your research and the entity's details, you must respond with EXACTLY ONE of the following XML-like tags at the very end of your response:

  <decision>APPROVE</decision>
  <decision>MERGE:TARGET_ID</decision>
  <decision>HUMAN_REVIEW</decision>
  """

  def new(entity, opts \\ []) do
    instruction = @system_prompt <> "\n" <> """
    ID: #{entity.entity_id}
    Name: #{entity.name}
    Type: #{entity.type}
    Description: #{entity.description}
    """

    base_opts = [
      name: "TriageAgent",
      instruction: instruction,
      planner: %ADK.Planner.PlanReAct{},
      tools: [ADK.Tool.GoogleSearch.new()]
    ]

    ADK.Agent.LlmAgent.new(Keyword.merge(base_opts, opts))
  end

  def run(entity, agent_opts \\ []) do
    agent = new(entity, agent_opts)
    runner = ADK.Runner.new(app_name: "triage", agent: agent)
    session_id = "triage_#{entity.id || entity.entity_id}"
    message = "Review this entity and decide its fate. Your output MUST end with a <decision> tag."
    events = ADK.Runner.run(runner, "system", session_id, message)
    
    # get text from the last agent_response event
    text = 
      events
      |> Enum.filter(&(&1.author == "TriageAgent"))
      |> Enum.map(&ADK.Event.text/1)
      |> Enum.reject(&is_nil/1)
      |> Enum.join("\n")

    parse_decision(text)
  end

  defp parse_decision(text) do
    case Regex.run(~r/<decision>(.*?)<\/decision>/, text) do
      [_, "APPROVE"] ->
        {:ok, "approved", nil}
      [_, "HUMAN_REVIEW"] ->
        {:ok, "needs_human", nil}
      [_, <<"MERGE:", target_id::binary>>] ->
        {:ok, "merged", String.trim(target_id)}
      _ ->
        {:error, :invalid_decision, text}
    end
  end
end
