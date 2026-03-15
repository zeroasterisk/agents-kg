defmodule AgentsKg.Parser.Worker do
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

      %Source{status: "processing", stage: "parse"} = source ->
        process_source(source)

      %Source{} = source ->
        Logger.info("Source #{id} already processed or not in processing/parse state: #{source.stage}/#{source.status}")
        :ok
    end
  end

  def process_source(%Source{} = source) do
    raw = source.raw_text || ""

    if raw == "" do
      Logger.error("No raw_text to parse for source #{source.id}")
      changeset = Source.changeset(source, %{error: "No raw_text to parse", status: "error"})
      Repo.update(changeset)
      {:error, "No raw_text to parse"}
    else
      source_type = source.type || "html"
      parsed = parse_content(raw, source_type)

      title = extract_title(parsed)

      changeset =
        Source.changeset(source, %{
          parsed_text: parsed,
          title: title,
          stage: "chunk",
          status: "processing"
        })

      case Repo.update(changeset) do
        {:ok, _} -> :ok
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp parse_content(raw, "pdf") do
    # PDF text from pymupdf — clean up whitespace, detect structure
    clean_whitespace(raw)
  end

  defp parse_content(raw, "html") do
    if is_markdown?(raw) do
      raw
    else
      html_to_text(raw)
    end
  end

  defp parse_content(raw, _other) do
    # markdown passthrough or unknown
    raw
  end

  defp clean_whitespace(text) do
    text
    |> String.replace(~r/\n{3,}/, "\n\n")
    |> String.trim()
  end

  defp is_markdown?(text) do
    Regex.match?(~r/^\#{1,6}\s/m, text)
  end

  defp html_to_text(html) do
    {title, summary} =
      try do
        doc = Floki.parse_document!(html)
        title =
          case Floki.find(doc, "title") do
            [] -> ""
            title_nodes -> Floki.text(title_nodes)
          end

        body =
          case Floki.find(doc, "body") do
            [] -> html
            [body_node | _] -> Floki.raw_html(body_node)
          end

        {title, body}
      rescue
        _ -> {"", html}
      end

    text = Regex.replace(~r/<[^>]+>/, summary, "\n")
    text = clean_whitespace(text)

    if title != "" and not String.starts_with?(text, title) do
      "# #{title}\n\n#{text}"
    else
      text
    end
  end

  defp extract_title(parsed) do
    case Regex.run(~r/^\#\s+(.+)/, parsed) do
      [_, title] -> String.trim(title)
      _ -> nil
    end
  end

  @doc """
  Enqueue jobs for all pending parse sources.
  """
  def enqueue_pending do
    import Ecto.Query

    Source
    |> where(status: "processing", stage: "parse")
    |> select([s], s.id)
    |> Repo.all()
    |> Enum.map(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
