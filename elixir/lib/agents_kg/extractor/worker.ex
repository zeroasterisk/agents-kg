defmodule AgentsKg.Extractor.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3
  
  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Entity
  alias AgentsKg.Edge
  alias AgentsKg.Extractor.Agent
  require Logger

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id} = args}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      %Source{status: "pending_extraction"} = source ->
        agent_opts = if args["mock"], do: [model: "mock"], else: []
        text = source.parsed_text || source.raw_text

        if is_nil(text) or text == "" do
          Logger.warning("Source #{id} has no text to extract from.")
          mark_source_complete(source, "skipped", "No text content")
          :ok
        else
          case Agent.run(text, "extract_#{id}", agent_opts) do
            {:ok, %{"entities" => entities, "edges" => edges}} ->
              Repo.transaction(fn ->
                Enum.each(entities || [], &insert_entity(&1, source.id))
                Enum.each(edges || [], &insert_edge(&1, source.id))
                mark_source_complete(source, "completed", nil)
              end)
              :ok

            {:error, reason} ->
              Logger.error("Extraction failed for source #{id}: #{inspect(reason)}")
              {:error, reason}
          end
        end

      source ->
        Logger.debug("Source #{id} in stage #{source.stage} and status #{source.status} - skipping extraction worker")
        :ok
    end
  end

  defp insert_entity(data, source_id) do
    aliases = if is_list(data["aliases"]), do: Jason.encode!(data["aliases"]), else: "[]"
    
    attrs = %{
      entity_id: data["entity_id"],
      name: data["name"],
      type: data["type"],
      kind: data["kind"],
      description: data["description"],
      aliases: aliases,
      source_id: source_id,
      status: "pending_review"
    }

    %Entity{}
    |> Entity.changeset(attrs)
    |> Repo.insert!(on_conflict: :nothing, conflict_target: :entity_id)
  end

  defp insert_edge(data, source_id) do
    edge_id = generate_edge_id(data)
    
    properties = if is_map(data["properties"]), do: Jason.encode!(data["properties"]), else: "{}"

    attrs = %{
      edge_id: edge_id,
      source_entity_id: data["source_entity_id"],
      target_entity_id: data["target_entity_id"],
      edge_type: data["edge_type"],
      properties: properties,
      confidence: (if is_number(data["confidence"]), do: data["confidence"], else: 0.5),
      source_id: source_id,
      status: "pending_review",
      extracted_at: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    %Edge{}
    |> Edge.changeset(attrs)
    |> Repo.insert!(on_conflict: :nothing, conflict_target: :edge_id)
  end

  defp generate_edge_id(data) do
    src = data["source_entity_id"] || ""
    tgt = data["target_entity_id"] || ""
    type = data["edge_type"] || ""
    raw = "#{src}|#{type}|#{tgt}"
    
    :sha256
    |> :crypto.hash(raw)
    |> Base.encode16(case: :lower)
    |> String.slice(0, 16)
  end

  defp mark_source_complete(source, status, error) do
    source
    |> Source.changeset(%{
      status: status,
      stage: "resolve",
      error: error
    })
    |> Repo.update!()
  end

  @doc """
  Enqueue jobs for all pending sources.
  """
  def enqueue_pending do
    import Ecto.Query
    
    Source
    |> where(status: "pending_extraction")
    |> select([s], s.id)
    |> Repo.all()
    |> Enum.map(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
