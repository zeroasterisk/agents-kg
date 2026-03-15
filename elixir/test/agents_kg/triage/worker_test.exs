defmodule AgentsKg.Triage.WorkerTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Entity
  alias AgentsKg.Triage.Worker

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)
    :ok
  end

  test "worker processes an entity and approves it" do
    ADK.LLM.Mock.set_responses(["<thought>Looks valid.</thought>\n<decision>APPROVE</decision>"])
    
    {:ok, entity} = Repo.insert(%Entity{
      entity_id: "ent-123",
      name: "Test Entity",
      type: "TestType",
      description: "A simple test entity.",
      status: "pending_review"
    })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => entity.id, "mock" => true}})

    updated = Repo.get(Entity, entity.id)
    assert updated.status == "approved"
    assert updated.merged_into == nil
  end

  test "worker processes an entity and marks for human review" do
    ADK.LLM.Mock.set_responses(["<thought>Not sure.</thought>\n<decision>HUMAN_REVIEW</decision>"])
    
    {:ok, entity} = Repo.insert(%Entity{
      entity_id: "ent-124",
      name: "Test Entity 2",
      type: "TestType",
      description: "Another test entity.",
      status: "pending_review"
    })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => entity.id, "mock" => true}})

    updated = Repo.get(Entity, entity.id)
    assert updated.status == "needs_human"
  end

  test "worker processes an entity and merges it" do
    ADK.LLM.Mock.set_responses(["<thought>Duplicate of ent-123.</thought>\n<decision>MERGE:ent-123</decision>"])
    
    {:ok, entity} = Repo.insert(%Entity{
      entity_id: "ent-125",
      name: "Test Entity 3",
      type: "TestType",
      description: "A duplicate test entity.",
      status: "pending_review"
    })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => entity.id, "mock" => true}})

    updated = Repo.get(Entity, entity.id)
    assert updated.status == "merged"
    assert updated.merged_into == "ent-123"
  end
end
