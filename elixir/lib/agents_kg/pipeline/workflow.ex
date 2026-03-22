defmodule AgentsKg.Pipeline.Workflow do
  @moduledoc """
  ADK.Workflow DAG for Agents-KG pipeline.
  """
  alias ADK.Workflow
  alias ADK.Agent.Custom

  def new do
    nodes = %{
      fetch:
        Custom.new(
          name: "fetch",
          handler: fn ctx ->
            source_id = ADK.Context.get_temp(ctx, :source_id)
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)
            AgentsKg.Fetcher.Worker.process_source(source)
            "fetch_complete"
          end
        ),
      parse:
        Custom.new(
          name: "parse",
          handler: fn ctx ->
            source_id = ADK.Context.get_temp(ctx, :source_id)
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)
            AgentsKg.Parser.Worker.process_source(source)
            "parse_complete"
          end
        ),
      chunk:
        Custom.new(
          name: "chunk",
          handler: fn ctx ->
            source_id = ADK.Context.get_temp(ctx, :source_id)
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)
            AgentsKg.Chunker.Worker.process_source(source)
            "chunk_complete"
          end
        ),
      extract: AgentsKg.Extractor.Agent.new(),
      qa: AgentsKg.Qa.Agent.new(),
      heal: AgentsKg.Heal.Agent.new(),
      human_review:
        Custom.new(
          name: "human_review",
          handler: fn _ctx ->
            # Stub for manual intervention
            "reviewed"
          end
        ),
      triage:
        Custom.new(
          name: "triage",
          handler: fn _ctx ->
            # Integration with AgentsKg.Triage.Agent
            # Note: triage usually works on entities, this might need restructuring 
            # for a source-level pipeline.
            "triage_complete"
          end
        ),
      load:
        Custom.new(
          name: "load",
          handler: fn ctx ->
            _source_id = ADK.Context.get_temp(ctx, :source_id)
            # Loader worker processes by ID
            # AgentsKg.Loader.Worker logic here
            "load_complete"
          end
        )
    }

    edges = [
      {:START, :fetch, :parse, :chunk, :extract, :qa},
      {:qa, %{"passed" => :triage, "failed" => :heal}},
      {:heal, %{"success" => :triage, "failure" => :human_review}},
      {:human_review, :triage},
      {:triage, :load, :END}
    ]

    Workflow.new(name: "agents_kg_pipeline", edges: edges, nodes: nodes)
  end
end
