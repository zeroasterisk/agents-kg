defmodule AgentsKg.Chunker.WorkerTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Chunk
  alias AgentsKg.Chunker.Worker
  import Ecto.Query

  @valid_html """
  Intro text goes here.

  # Section 1
  This is the first section.
  It has some text.

  ## Section 1.1
  A subsection.
  """

  # Generate long text that exceeds tokens
  @long_text "Intro\n\n# Long Section\n" <>
               String.duplicate("A paragraph that is repeated to make it long.\n\n", 100)

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)

    source =
      Repo.insert!(%Source{
        uri: "https://example.com/test",
        type: "html",
        stage: "chunk",
        status: "processing",
        parsed_text: @valid_html
      })

    %{source: source}
  end

  describe "perform/1" do
    test "processes source and creates chunks", %{source: source} do
      job = %Oban.Job{args: %{"id" => source.id}}

      assert :ok = Worker.perform(job)

      # Reload source
      updated_source = Repo.get(Source, source.id)
      assert updated_source.stage == "extract"
      assert updated_source.status == "pending_extraction"

      # Check chunks
      chunks = Repo.all(from(c in Chunk, where: c.source_id == ^source.id, order_by: c.position))

      assert length(chunks) == 3

      [c1, c2, c3] = chunks

      assert c1.position == 0
      assert c1.section_heading == nil
      assert c1.text == "Intro text goes here."

      assert c2.position == 1
      assert c2.section_heading == "# Section 1"
      assert c2.text == "# Section 1\n\nThis is the first section.\nIt has some text."

      assert c3.position == 2
      assert c3.section_heading == "## Section 1.1"
      assert c3.text == "## Section 1.1\n\nA subsection."
    end

    test "splits long sections into multiple chunks" do
      source =
        Repo.insert!(%Source{
          uri: "https://example.com/long",
          type: "text",
          stage: "chunk",
          status: "processing",
          parsed_text: @long_text
        })

      job = %Oban.Job{args: %{"id" => source.id}}
      assert :ok = Worker.perform(job)

      chunks = Repo.all(from(c in Chunk, where: c.source_id == ^source.id, order_by: c.position))

      # There should be more than 2 chunks (1 for Intro, and multiple for the long section)
      assert length(chunks) > 2

      # Verify chunks are properly numbered
      positions = Enum.map(chunks, & &1.position)
      assert positions == Enum.to_list(0..(length(chunks) - 1))

      # Verify token counts are reasonable
      Enum.each(chunks, fn chunk ->
        # Roughly target tokens + max tokens boundary check
        assert chunk.token_count <= 800
      end)
    end

    test "handles already processed source gracefully", %{source: source} do
      # Set to a different stage
      changeset = Source.changeset(source, %{stage: "extract", status: "pending_extraction"})
      Repo.update!(changeset)

      job = %Oban.Job{args: %{"id" => source.id}}
      assert :ok = Worker.perform(job)

      # Should not have created chunks
      chunks = Repo.all(from(c in Chunk, where: c.source_id == ^source.id))
      assert chunks == []
    end

    test "returns error when source not found" do
      job = %Oban.Job{args: %{"id" => 999_999}}
      assert {:error, :not_found} = Worker.perform(job)
    end

    test "returns error when no text is available" do
      source =
        Repo.insert!(%Source{
          uri: "https://example.com/empty",
          type: "html",
          stage: "chunk",
          status: "processing",
          parsed_text: nil,
          raw_text: ""
        })

      job = %Oban.Job{args: %{"id" => source.id}}
      assert {:error, "No text to chunk"} = Worker.perform(job)

      updated_source = Repo.get(Source, source.id)
      assert updated_source.status == "error"
      assert updated_source.error == "No text to chunk"
    end
  end
end
