defmodule AgentsKg.Triage.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3

  alias AgentsKg.Repo
  alias AgentsKg.Entity
  alias AgentsKg.Triage.Agent

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id} = args}) do
    case Repo.get(Entity, id) do
      nil ->
        {:error, :not_found}

      %Entity{status: "pending_review"} = entity ->
        agent_opts = if args["mock"], do: [model: "mock"], else: []

        case Agent.run(entity, agent_opts) do
          {:ok, status, merged_into} ->
            changeset = Entity.changeset(entity, %{status: status, merged_into: merged_into})

            case Repo.update(changeset) do
              {:ok, entity} ->
                # Re-trigger orchestrator for the source
                Oban.insert(AgentsKg.Pipeline.Orchestrator.new(%{"id" => entity.source_id}))
                :ok

              {:error, changeset} ->
                {:error, changeset}
            end

          {:error, reason, text} ->
            _ = reason
            _ = text
            require Logger

            Logger.warning(
              "Failed to parse decision for entity #{id}. Reason: #{inspect(reason)}. Text: #{inspect(text)}"
            )

            {:error, :parse_failure}
        end

      _entity ->
        # Already processed or not pending review
        :ok
    end
  end

  @doc """
  Enqueue jobs for all pending entities.
  """
  def enqueue_pending do
    import Ecto.Query

    Entity
    |> where(status: "pending_review")
    |> select([e], e.id)
    |> Repo.all()
    |> Enum.map(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
