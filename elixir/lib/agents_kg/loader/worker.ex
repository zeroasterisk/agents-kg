defmodule AgentsKg.Loader.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Entity
  alias AgentsKg.Edge
  import Ecto.Query
  require Logger

  @yaml_dir Path.join(File.cwd!(), "../kg/entities")

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id}}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      %Source{stage: "load"} = source ->
        Logger.info("Running loader for source #{id}")
        # 1. Fetch approved entities and edges
        entities =
          Repo.all(from(e in Entity, where: e.source_id == ^id and e.status == "approved"))

        edges = Repo.all(from(e in Edge, where: e.source_id == ^id and e.status == "approved"))

        if entities == [] and edges == [] do
          Logger.info("No approved items to load for source #{id}")
          mark_source_complete(source, "complete", "done", nil)
          :ok
        else
          # 2. Export YAML
          Enum.each(entities, &export_yaml/1)

          # 3. Neo4j load
          load_to_neo4j(entities, edges)

          # 4. Update source
          mark_source_complete(source, "complete", "done", nil)
          :ok
        end

      source ->
        Logger.debug(
          "Source #{id} in stage #{source.stage} and status #{source.status} - skipping loader worker"
        )

        :ok
    end
  end

  defp export_yaml(entity) do
    etype = String.downcase(entity.type || "unknown")
    dir_path = Path.join(@yaml_dir, "#{etype}s")
    File.mkdir_p!(dir_path)

    eid =
      if String.contains?(entity.entity_id, ":") do
        List.last(String.split(entity.entity_id, ":", parts: 2))
      else
        entity.entity_id
      end

    file_path = Path.join(dir_path, "#{eid}.yaml")

    aliases =
      case Jason.decode(entity.aliases || "[]") do
        {:ok, decoded} -> decoded
        _ -> []
      end

    data = %{
      "id" => entity.entity_id,
      "name" => entity.name,
      "type" => entity.type,
      "kind" => entity.kind,
      "description" => entity.description,
      "aliases" => aliases
    }

    # Remove nil values
    data = Enum.reject(data, fn {_, v} -> is_nil(v) end) |> Enum.into(%{})

    yaml_string = encode_yaml(data)

    File.write!(file_path, yaml_string)
    Logger.info("Exported #{file_path}")
  end

  defp encode_yaml(map) do
    Enum.map_join(map, "\n", fn {k, v} ->
      if is_list(v) do
        "#{k}:\n" <> Enum.map_join(v, "\n", fn item -> "  - #{inspect(item)}" end)
      else
        "#{k}: #{inspect(v)}"
      end
    end) <> "\n"
  end

  defp load_to_neo4j(entities, edges) do
    uri = System.get_env("NEO4J_URI")
    user = System.get_env("NEO4J_USER", "neo4j")
    password = System.get_env("NEO4J_PASSWORD")

    if uri && password do
      Logger.info("Loading #{length(entities)} entities, #{length(edges)} edges to Neo4j")

      auth = Base.encode64("#{user}:#{password}")

      headers = [
        {"Authorization", "Basic #{auth}"},
        {"Content-Type", "application/json"},
        {"Accept", "application/json"}
      ]

      db_url = "#{uri}/db/neo4j/tx/commit"

      # Process entities
      entity_statements = Enum.map(entities, &entity_to_statement/1)

      if entity_statements != [] do
        case Req.post(db_url, headers: headers, json: %{statements: entity_statements}) do
          {:ok, %{status: 200, body: %{"errors" => []}}} ->
            Logger.info("Successfully loaded entities to Neo4j")

          {:ok, %{body: body}} ->
            Logger.error("Neo4j load failed for entities: #{inspect(body)}")

          error ->
            Logger.error("Neo4j request failed for entities: #{inspect(error)}")
        end
      end

      # Process edges
      edge_statements = Enum.map(edges, &edge_to_statement/1)

      if edge_statements != [] do
        case Req.post(db_url, headers: headers, json: %{statements: edge_statements}) do
          {:ok, %{status: 200, body: %{"errors" => []}}} ->
            Logger.info("Successfully loaded edges to Neo4j")

          {:ok, %{body: body}} ->
            Logger.error("Neo4j load failed for edges: #{inspect(body)}")

          error ->
            Logger.error("Neo4j request failed for edges: #{inspect(error)}")
        end
      end
    else
      Logger.info(
        "Neo4j not configured (NEO4J_URI and NEO4J_PASSWORD required), skipping graph load"
      )
    end
  end

  defp entity_to_statement(entity) do
    aliases =
      case Jason.decode(entity.aliases || "[]") do
        {:ok, decoded} -> decoded
        _ -> []
      end

    params = %{
      "entity_id" => entity.entity_id,
      "name" => entity.name,
      "type" => entity.type,
      "kind" => entity.kind,
      "description" => entity.description,
      "aliases" => aliases
    }

    query = """
    MERGE (n {entity_id: $entity_id})
    SET n:Entity, n.name = $name, n.type = $type, n.kind = $kind,
        n.description = $description, n.aliases = $aliases
    """

    %{statement: query, parameters: params}
  end

  defp edge_to_statement(edge) do
    props =
      case Jason.decode(edge.properties || "{}") do
        {:ok, decoded} -> decoded
        _ -> %{}
      end

    params = %{
      "src" => edge.source_entity_id,
      "tgt" => edge.target_entity_id,
      "edge_id" => edge.edge_id,
      "confidence" => edge.confidence,
      "source_type" => edge.source_type
    }

    params = Enum.reduce(props, params, fn {k, v}, acc -> Map.put(acc, "prop_#{k}", v) end)

    edge_type = String.upcase(edge.edge_type || "RELATED")

    prop_sets = Enum.map_join(props, ", ", fn {k, _} -> "r.#{k} = $prop_#{k}" end)
    extra = if prop_sets == "", do: "", else: ", #{prop_sets}"

    query = """
    MATCH (a {entity_id: $src}), (b {entity_id: $tgt})
    MERGE (a)-[r:#{edge_type} {edge_id: $edge_id}]->(b)
    SET r.confidence = $confidence, r.source_type = $source_type#{extra}
    """

    %{statement: query, parameters: params}
  end

  defp mark_source_complete(source, status, stage, error) do
    source
    |> Source.changeset(%{
      status: status,
      stage: stage,
      error: error
    })
    |> Repo.update!()
  end

  @doc """
  Enqueue jobs for all pending sources in load stage.
  """
  def enqueue_pending do
    Source
    |> where(stage: "load")
    |> select([s], s.id)
    |> Repo.all()
    |> Enum.map(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
