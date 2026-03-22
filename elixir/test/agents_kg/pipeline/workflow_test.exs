defmodule AgentsKg.Pipeline.WorkflowTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Pipeline.Workflow

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

  test "runs the complete workflow from fetch to extract", %{temp_file: temp_file} do
    {:ok, source_file} =
      AgentsKg.Repo.insert(%Source{
        uri: "file://#{temp_file}",
        status: "pending",
        stage: "fetch",
        max_attempts: 3,
        attempts: 0,
        type: "text"
      })

    workflow = Workflow.new(source_file.id)
    ctx = %ADK.Context{invocation_id: "test_session_#{source_file.id}"}

    events = ADK.Workflow.run(workflow, ctx)

    authors = Enum.map(events, & &1.author)

    assert "fetch" in authors
    assert "parse" in authors
    assert "chunk" in authors
    assert "extract" in authors
  end
end
