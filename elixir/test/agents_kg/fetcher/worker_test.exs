defmodule AgentsKg.Fetcher.WorkerTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Fetcher.Worker

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)

    # Create a dummy file
    temp_file = Path.join(System.tmp_dir!(), "test_source_#{System.unique_integer()}.txt")
    File.write!(temp_file, "This is a test content")

    on_exit(fn ->
      File.rm(temp_file)
    end)

    {:ok, %{temp_file: temp_file}}
  end

  test "worker fetches a local file and updates the source", %{temp_file: temp_file} do
    {:ok, source} =
      Repo.insert(%Source{
        uri: "file://" <> temp_file,
        status: "pending",
        stage: "fetch"
      })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => source.id}})

    updated = Repo.get(Source, source.id)
    assert updated.status == "processing"
    assert updated.stage == "parse"
    assert updated.raw_text == "This is a test content"
    assert updated.type == "text"

    assert updated.content_hash ==
             "985bee5cee8b11457985415cb3864ddb04e167f9ade692af9ad859ffb6e2d8ca"
  end

  test "worker skips fetching if content hash matches", %{temp_file: temp_file} do
    # Pre-compute the hash
    hash = :crypto.hash(:sha256, "This is a test content") |> Base.encode16(case: :lower)

    {:ok, source} =
      Repo.insert(%Source{
        uri: "file://" <> temp_file,
        status: "pending",
        stage: "fetch",
        content_hash: hash
      })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => source.id}})

    updated = Repo.get(Source, source.id)
    # When unchanged, it's marked as complete/done
    assert updated.status == "complete"
    assert updated.stage == "done"
    assert updated.raw_text == nil
  end

  test "worker records an error when fetching fails" do
    {:ok, source} =
      Repo.insert(%Source{
        uri: "file:///does_not_exist.txt",
        status: "pending",
        stage: "fetch"
      })

    assert {:error, _} = Worker.perform(%Oban.Job{args: %{"id" => source.id}})

    updated = Repo.get(Source, source.id)
    assert updated.status == "error"
    assert updated.error =~ "File not found"
  end

  test "worker safely returns :ok when source doesn't exist or isn't pending" do
    assert {:error, :not_found} == Worker.perform(%Oban.Job{args: %{"id" => -1}})

    {:ok, source} =
      Repo.insert(%Source{
        uri: "http://example.com",
        status: "processing",
        stage: "parse"
      })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => source.id}})
  end

  test "enqueue_pending/0 queues jobs for pending sources" do
    {:ok, source} =
      Repo.insert(%Source{
        uri: "http://example.com/pending",
        status: "pending",
        stage: "fetch"
      })

    {:ok, _other} =
      Repo.insert(%Source{
        uri: "http://example.com/done",
        status: "complete",
        stage: "done"
      })

    jobs = Worker.enqueue_pending()
    assert length(jobs) == 1

    [{:ok, %{args: %{"id" => id}}}] = jobs
    assert id == source.id
  end
end
