defmodule AgentsKg.Extractor.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3
  
  import Ecto.Query
  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Chunk
  alias AgentsKg.Entity
  alias AgentsKg.Edge
  alias AgentsKg.Extractor.Agent
  require Logger

  @valid_entity_types MapSet.new([
    "Organization", "Group", "Person", "Project", "Protocol", "Capability"
  ])

  @valid_edge_types MapSet.new([
    "MEMBER_OF", "GOVERNS", "DEVELOPS", "IMPLEMENTS", "COMPETES_WITH",
    "ADDRESSES", "AUTHORED", "CHAIRS", "SPONSORS", "PART_OF",
    "SUPERSEDES", "CONTRIBUTES_TO", "DEFINES", "COMPLEMENTS", "USES"
  ])

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id} = args}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      %Source{status: "pending_extraction"} = source ->
        agent_opts = if args["mock"], do: [model: "mock"], else: []
        
        chunks = Repo.all(from c in Chunk, where: c.source_id == ^id, order_by: c.position)

        if chunks == [] do
          Logger.warning("Source #{id} has no chunks to extract from.")
          mark_source_complete(source, "skipped", "No chunks to extract from")
          :ok
        else
          results = 
            Enum.map(chunks, fn chunk ->
              Logger.info("Extracting from chunk #{chunk.id} (source #{id})")
              case Agent.run(chunk.text, "extract_#{id}_#{chunk.id}", agent_opts) do
                {:ok, data} -> {:ok, data, chunk}
                {:error, reason} -> {:error, reason, chunk}
              end
            end)
          
          # Check if all chunks failed
          failures = Enum.filter(results, fn {status, _, _} -> status == :error end)
          successes = Enum.filter(results, fn {status, _, _} -> status == :ok end)
          
          if length(successes) == 0 and length(failures) > 0 do
            {:error, "All chunk extractions failed"}
          else
            Repo.transaction(fn ->
              for {:ok, %{"entities" => entities, "edges" => edges}, chunk} <- successes do
                Enum.each(entities || [], &insert_entity(&1, source.id, chunk.id))
                Enum.each(edges || [], &insert_edge(&1, source.id, chunk.id))
              end
              mark_source_complete(source, "processing", nil) # Move to 'processing' for triage per Python script
            end)
            :ok
          end
        end

      source ->
        Logger.debug("Source #{id} in stage #{source.stage} and status #{source.status} - skipping extraction worker")
        :ok
    end
  end

  defp insert_entity(data, source_id, chunk_id) do
    type = data["type"] || ""
    if not MapSet.member?(@valid_entity_types, type) do
      Logger.warning("Skipping entity with invalid type #{inspect(type)}: #{data["entity_id"]}")
    else
      aliases = if is_list(data["aliases"]), do: Jason.encode!(data["aliases"]), else: "[]"
      
      attrs = %{
        entity_id: data["entity_id"],
        name: data["name"],
        type: type,
        kind: data["kind"],
        description: data["description"],
        aliases: aliases,
        source_id: source_id,
        chunk_id: chunk_id,
        status: "pending_review"
      }

      %Entity{}
      |> Entity.changeset(attrs)
      |> Repo.insert!(on_conflict: :nothing, conflict_target: :entity_id)
    end
  end

  defp insert_edge(data, source_id, chunk_id) do
    edge_type = data["edge_type"] || ""
    if not MapSet.member?(@valid_edge_types, edge_type) do
      Logger.warning("Skipping edge with invalid type #{inspect(edge_type)}: #{data["source_entity_id"]} -> #{data["target_entity_id"]}")
    else
      edge_id = generate_edge_id(data)
      
      properties = if is_map(data["properties"]), do: Jason.encode!(data["properties"]), else: "{}"

      attrs = %{
        edge_id: edge_id,
        source_entity_id: data["source_entity_id"],
        target_entity_id: data["target_entity_id"],
        edge_type: edge_type,
        properties: properties,
        confidence: (if is_number(data["confidence"]), do: data["confidence"], else: 0.5),
        source_id: source_id,
        chunk_id: chunk_id,
        status: "pending_review",
        extracted_at: DateTime.utc_now() |> DateTime.to_iso8601()
      }

      %Edge{}
      |> Edge.changeset(attrs)
      |> Repo.insert!(on_conflict: :nothing, conflict_target: :edge_id)
    end
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
