defmodule AgentsKg.Resolver.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3

  import Ecto.Query
  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Entity
  require Logger

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id}}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      %Source{status: "processing", stage: "resolve"} = source ->
        Logger.info("Resolving entities for source #{id}")

        # Enqueue Triage worker for all pending entities
        Entity
        |> where([e], e.source_id == ^id and e.status == "pending_review")
        |> select([e], e.id)
        |> Repo.all()
        |> Enum.each(fn entity_id ->
          %{"id" => entity_id}
          |> AgentsKg.Triage.Worker.new()
          |> Oban.insert!()
        end)

        # Update source to review stage
        updated_source =
          source
          |> Source.changeset(%{stage: "review", status: "pending_review"})
          |> Repo.update!()

        # Re-trigger orchestrator to check review status
        Oban.insert(AgentsKg.Pipeline.Orchestrator.new(%{"id" => source.id}))

        :ok

      source ->
        Logger.debug("Source #{id} not in resolve stage: #{source.stage}/#{source.status}")
        :ok
    end
  end
end
