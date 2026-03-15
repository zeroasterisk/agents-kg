defmodule AgentsKg.Chunker.Worker do
  use Oban.Worker, queue: :default, max_attempts: 3

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Chunk
  import Ecto.Query
  require Logger

  @target_tokens 500
  @max_tokens 800

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"id" => id}}) do
    case Repo.get(Source, id) do
      nil ->
        {:error, :not_found}

      %Source{status: "processing", stage: "chunk"} = source ->
        process_source(source)

      %Source{} = source ->
        Logger.info(
          "Source #{id} already processed or not in processing/chunk state: #{source.stage}/#{source.status}"
        )

        :ok
    end
  end

  def process_source(%Source{} = source) do
    text = source.parsed_text || source.raw_text || ""

    if text == "" do
      Logger.error("No text to chunk for source #{source.id}")
      changeset = Source.changeset(source, %{error: "No text to chunk", status: "error"})
      Repo.update(changeset)
      {:error, "No text to chunk"}
    else
      # delete existing chunks for source
      from(c in Chunk, where: c.source_id == ^source.id) |> Repo.delete_all()

      sections = split_sections(text)

      chunks_data =
        sections
        |> Enum.flat_map(fn {heading, body} ->
          if estimate_tokens(body) > @max_tokens do
            split_long(body, heading)
          else
            [{heading, body}]
          end
        end)
        |> Enum.with_index()
        |> Enum.map(fn {{heading, chunk_text}, position} ->
          full_text =
            if heading != "" and heading != nil do
              "#{heading}\n\n#{chunk_text}" |> String.trim()
            else
              chunk_text
            end

          tokens = estimate_tokens(full_text)

          %{
            source_id: source.id,
            text: full_text,
            position: position,
            section_heading: if(heading != "", do: heading, else: nil),
            token_count: tokens,
            chunk_strategy: "section"
          }
        end)

      # Insert chunks
      Enum.each(chunks_data, fn attrs ->
        %Chunk{}
        |> Chunk.changeset(attrs)
        |> Repo.insert!()
      end)

      Logger.info("Created #{length(chunks_data)} chunks for source #{source.id}")

      changeset =
        Source.changeset(source, %{
          stage: "extract",
          status: "pending_extraction"
        })

      case Repo.update(changeset) do
        {:ok, _} -> :ok
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp estimate_tokens(text) do
    div(String.length(text), 4)
  end

  defp split_sections(text) do
    parts = Regex.split(~r/^(\#{1,6}\s+.+)$/m, text, include_captures: true)

    {sections, final_heading, final_body} =
      Enum.reduce(parts, {[], "", ""}, fn part, {acc, current_heading, current_body} ->
        if Regex.match?(~r/^\#{1,6}\s+/, part) do
          new_acc =
            if String.trim(current_body) != "" do
              acc ++ [{current_heading, String.trim(current_body)}]
            else
              acc
            end

          {new_acc, String.trim(part), ""}
        else
          {acc, current_heading, current_body <> part}
        end
      end)

    final =
      if String.trim(final_body) != "" do
        sections ++ [{final_heading, String.trim(final_body)}]
      else
        sections
      end

    if final == [] do
      [{"", text}]
    else
      final
    end
  end

  defp split_long(text, heading) do
    paragraphs = Regex.split(~r/\n\n+/, text)

    {chunks, current} =
      Enum.reduce(paragraphs, {[], ""}, fn para, {acc, current} ->
        combined = if current == "", do: para, else: current <> "\n\n" <> para

        if estimate_tokens(combined) > @target_tokens and current != "" do
          {acc ++ [{heading, String.trim(current)}], para}
        else
          {acc, String.trim(combined)}
        end
      end)

    if String.trim(current) != "" do
      chunks ++ [{heading, String.trim(current)}]
    else
      chunks
    end
  end

  @doc """
  Enqueue jobs for all pending chunk sources.
  """
  def enqueue_pending do
    import Ecto.Query

    Source
    |> where(status: "processing", stage: "chunk")
    |> select([s], s.id)
    |> Repo.all()
    |> Enum.map(fn id ->
      %{"id" => id}
      |> __MODULE__.new()
      |> Oban.insert()
    end)
  end
end
