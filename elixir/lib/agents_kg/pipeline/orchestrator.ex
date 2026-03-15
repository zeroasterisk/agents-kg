defmodule AgentsKg.Pipeline.Orchestrator do
  @moduledoc """
  Replaces Prefect's `process_source` flow. 
  Takes a source_id, determines its current stage/status, and enqueues the appropriate worker.
  """
  use Oban.Worker, queue: :default

  alias AgentsKg.Repo
  alias AgentsKg.Source
  require Logger

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id}}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      source ->
        cond do
          source.status in ["complete", "failed", "dead_letter", "error"] ->
            Logger.info("Source #{id} pipeline complete: #{source.stage}/#{source.status}")
            :ok

          source.stage == "review" or source.status == "pending_review" ->
            Logger.info("Source #{id} waiting for review. Checking if entities are resolved...")
            check_and_advance_review(source)
            :ok

          true ->
            dispatch_stage(source)
        end
    end
  end

  defp check_and_advance_review(source) do
    import Ecto.Query
    alias AgentsKg.Entity

    unresolved_count =
      Entity
      |> where([e], e.source_id == ^source.id and e.status in ["pending_review", "needs_human"])
      |> Repo.aggregate(:count, :id)

    if unresolved_count == 0 do
      Logger.info("All entities for source #{source.id} resolved. Advancing to load stage.")
      
      updated_source =
        source
        |> Source.changeset(%{stage: "load", status: "processing"})
        |> Repo.update!()

      dispatch_stage(updated_source)
    else
      Logger.info("Source #{source.id} has #{unresolved_count} unresolved entities. Waiting.")
    end
  end

  defp dispatch_stage(source) do
    Logger.info("Orchestrating source #{source.id} at stage: #{source.stage}")

    worker =
      case source.stage do
        "fetch" -> AgentsKg.Fetcher.Worker
        "parse" -> AgentsKg.Parser.Worker
        "chunk" -> AgentsKg.Chunker.Worker
        "extract" -> AgentsKg.Extractor.Worker
        "resolve" -> AgentsKg.Resolver.Worker
        "load" -> AgentsKg.Loader.Worker
        "done" -> nil
        "review" -> nil
        _ -> nil
      end

    if worker do
      %{id: source.id}
      |> worker.new()
      |> Oban.insert!()
      :ok
    else
      if source.stage not in ["review", "done"] do
        Logger.error("No worker defined for stage #{source.stage}")
      end
      :ok
    end
  end

  @doc """
  Replaces Prefect's `process_all_sources` loop.
  Enqueues an orchestrator job for all sources not in a terminal state.
  """
  def enqueue_all_pending do
    import Ecto.Query

    Source
    |> where([s], s.status not in ["complete", "failed", "dead_letter", "error", "pending_review"])
    |> where([s], s.stage not in ["done", "review"])
    |> select([s], s.id)
    |> Repo.all()
    |> Enum.each(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
