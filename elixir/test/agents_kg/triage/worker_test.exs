defmodule AgentsKg.Triage.WorkerTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Entity
  alias AgentsKg.Triage.Worker

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)
    ADK.LLM.Mock.set_responses(["<thought>Looks valid.</thought>\n<decision>APPROVE</decision>"])
    :ok
  end

  test "worker processes an entity and approves it" do
    {:ok, entity} = Repo.insert(%Entity{
      entity_id: "ent-123",
      name: "Test Entity",
      type: "TestType",
      description: "A simple test entity.",
      status: "pending_review"
    })

    assert entity.status == "pending_review"

    # Run the worker inline
    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => entity.id, "mock" => true}})

    # Check updated entity
    updated = Repo.get(Entity, entity.id)
    assert updated.status == "approved"
  end
end
