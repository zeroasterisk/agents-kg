defmodule AgentsKg.Pipeline.Workflow do
  @moduledoc """
  ADK.Workflow DAG for Agents-KG pipeline.
  """
  alias ADK.Workflow
  alias ADK.Agent.Custom
  require Logger

  def new(source_id) do
    nodes = %{
      fetch:
        Custom.new(
          name: "fetch",
          run_fn: fn _agent, _ctx ->
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)

            case AgentsKg.Fetcher.Worker.process_source(source) do
              :ok ->
                Logger.info("Workflow: fetch completed for source #{source_id}")
                [ADK.Event.new(%{author: "fetch", content: "fetch_complete"})]

              {:error, reason} ->
                Logger.error("Workflow: fetch failed for source #{source_id}: #{inspect(reason)}")
                [ADK.Event.error(inspect(reason), author: "fetch")]
            end
          end
        ),
      parse:
        Custom.new(
          name: "parse",
          run_fn: fn _agent, _ctx ->
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)

            case AgentsKg.Parser.Worker.process_source(source) do
              :ok ->
                Logger.info("Workflow: parse completed for source #{source_id}")
                [ADK.Event.new(%{author: "parse", content: "parse_complete"})]

              {:error, reason} ->
                Logger.error("Workflow: parse failed for source #{source_id}: #{inspect(reason)}")
                [ADK.Event.error(inspect(reason), author: "parse")]
            end
          end
        ),
      chunk:
        Custom.new(
          name: "chunk",
          run_fn: fn _agent, _ctx ->
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)

            case AgentsKg.Chunker.Worker.process_source(source) do
              :ok ->
                Logger.info("Workflow: chunk completed for source #{source_id}")
                [ADK.Event.new(%{author: "chunk", content: "chunk_complete"})]

              {:error, reason} ->
                Logger.error("Workflow: chunk failed for source #{source_id}: #{inspect(reason)}")
                [ADK.Event.error(inspect(reason), author: "chunk")]
            end
          end
        ),
      extract:
        Custom.new(
          name: "extract",
          run_fn: fn _agent, _ctx ->
            source = AgentsKg.Repo.get!(AgentsKg.Source, source_id)
            # Extractor.Worker returns :ok or {:error, reason}
            case AgentsKg.Extractor.Worker.process_source(source) do
              :ok ->
                Logger.info("Workflow: extract completed for source #{source_id}")
                [ADK.Event.new(%{author: "extract", content: "extract_complete"})]

              {:error, reason} ->
                Logger.error(
                  "Workflow: extract failed for source #{source_id}: #{inspect(reason)}"
                )

                [ADK.Event.error(inspect(reason), author: "extract")]
            end
          end
        )
    }

    edges = [
      {:START, :fetch, :parse, :chunk, :extract, :END}
    ]

    Workflow.new(name: "agents_kg_pipeline_#{source_id}", edges: edges, nodes: nodes)
  end
end
