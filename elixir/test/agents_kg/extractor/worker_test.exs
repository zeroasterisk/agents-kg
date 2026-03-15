defmodule AgentsKg.Extractor.WorkerTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Entity
  alias AgentsKg.Edge
  alias AgentsKg.Extractor.Worker

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)
    :ok
  end

  test "worker extracts entities and edges from a source" do
    unique_id = "organization:google-test-#{System.unique_integer([:positive])}"
    mock_response = %{
      "entities" => [
        %{
          "entity_id" => unique_id,
          "name" => "Google",
          "type" => "Organization",
          "kind" => "company",
          "description" => "Search engine company",
          "aliases" => ["Alphabet"]
        }
      ],
      "edges" => [
        %{
          "source_entity_id" => unique_id,
          "target_entity_id" => "project:vertex-ai",
          "edge_type" => "DEVELOPS",
          "confidence" => 0.9,
          "properties" => %{}
        }
      ]
    }
    
    ADK.LLM.Mock.set_responses([Jason.encode!(mock_response)])
    
    {:ok, source} = Repo.insert(%Source{
      uri: "https://example.com/test",
      raw_text: "Google develops Vertex AI.",
      status: "pending_extraction",
      stage: "extract"
    })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => source.id, "mock" => true}})

    # Check entities
    entity = Repo.get_by(Entity, entity_id: unique_id, source_id: source.id)
    assert entity
    assert entity.name == "Google"
    assert entity.status == "pending_review"

    # Check edges
    edge = Repo.get_by(Edge, source_entity_id: unique_id, source_id: source.id)
    assert edge
    assert edge.target_entity_id == "project:vertex-ai"
    assert edge.edge_type == "DEVELOPS"
    assert edge.status == "pending_review"

    # Check source status
    updated_source = Repo.get(Source, source.id)
    assert updated_source.status == "completed"
    assert updated_source.stage == "resolve"
  end
end
