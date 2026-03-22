defmodule AgentsKg.Fetcher.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3

  alias AgentsKg.Repo
  alias AgentsKg.Source
  import Ecto.Query
  require Logger

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id}}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      %Source{status: "pending", stage: "fetch"} = source ->
        process_source(source)

      %Source{} = source ->
        Logger.info(
          "Source #{id} already processed or not in pending/fetch state: #{source.stage}/#{source.status}"
        )

        :ok
    end
  end

  def process_source(%Source{} = source) do
    uri = source.uri
    Logger.info("Fetching #{uri}")

    case fetch_content(uri) do
      {:ok, raw_text, source_type} ->
        new_hash = :crypto.hash(:sha256, raw_text) |> Base.encode16(case: :lower)

        if source.content_hash == new_hash do
          Logger.info("Content unchanged for #{uri}, skipping")

          changeset = Source.changeset(source, %{status: "complete", stage: "done"})

          case Repo.update(changeset) do
            {:ok, _} ->
              :ok

            {:error, reason} ->
              {:error, reason}
          end
        else
          changeset =
            Source.changeset(source, %{
              raw_text: raw_text,
              content_hash: new_hash,
              type: source_type,
              stage: "parse",
              status: "processing"
            })

          case Repo.update(changeset) do
            {:ok, _} ->
              :ok

            {:error, reason} ->
              {:error, reason}
          end
        end

      {:error, reason} ->
        Logger.error("Failed to fetch #{uri}: #{inspect(reason)}")

        reason_str = if is_binary(reason), do: reason, else: inspect(reason)

        changeset = Source.changeset(source, %{error: reason_str, status: "error"})
        Repo.update(changeset)

        {:error, reason}
    end
  end

  defp fetch_content("file://" <> path) do
    do_fetch_local(path)
  end

  defp fetch_content(uri) do
    if is_local_file?(uri) do
      do_fetch_local(uri)
    else
      do_fetch_remote(uri)
    end
  end

  defp is_local_file?(uri) do
    try do
      File.regular?(Path.expand(uri))
    rescue
      _ -> false
    end
  end

  defp do_fetch_local(path) do
    abs_path = Path.expand(path)

    if not File.regular?(abs_path) do
      {:error, "File not found: #{abs_path}"}
    else
      if String.ends_with?(String.downcase(abs_path), ".pdf") do
        # Emulate python behavior which requires pymupdf
        {:error, "pymupdf not installed natively. Can't parse PDF"}
      else
        case File.read(abs_path) do
          {:ok, content} -> {:ok, content, "text"}
          {:error, reason} -> {:error, inspect(reason)}
        end
      end
    end
  end

  defp do_fetch_remote(uri) do
    # Requires HTTP fetching
    case Req.get(uri, redirect: true, max_retries: 3) do
      {:ok, %Req.Response{status: status} = resp} when status in 200..299 ->
        content_type_header = Req.Response.get_header(resp, "content-type")
        content_type = List.first(content_type_header) || ""

        source_type = if String.contains?(content_type, "html"), do: "html", else: "text"

        body =
          if is_binary(resp.body) do
            resp.body
          else
            Jason.encode!(resp.body)
          end

        {:ok, body, source_type}

      {:ok, %Req.Response{status: status}} ->
        {:error, "HTTP error: #{status}"}

      {:error, %{reason: reason}} ->
        {:error, inspect(reason)}

      {:error, reason} ->
        {:error, inspect(reason)}
    end
  end

  @doc """
  Enqueue jobs for all pending fetch sources.
  """
  def enqueue_pending do
    import Ecto.Query

    Source
    |> where(status: "pending", stage: "fetch")
    |> select([s], s.id)
    |> Repo.all()
    |> Enum.map(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
