defmodule AgentsKg.Extractor.WorkerTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Repo
  alias AgentsKg.Source
  alias AgentsKg.Chunk
  alias AgentsKg.Entity
  alias AgentsKg.Edge
  alias AgentsKg.Extractor.Worker

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)
    :ok
  end

  test "worker extracts entities and edges from source chunks" do
    mock_response = %{
      "entities" => [
        %{
          "entity_id" => "organization:google-worker-test-123",
          "name" => "Google",
          "type" => "Organization",
          "kind" => "company",
          "description" => "Search engine company",
          "aliases" => ["Alphabet"]
        },
        %{
          "entity_id" => "invalid:test",
          "name" => "Invalid",
          "type" => "InvalidType",
          "description" => "Should be skipped"
        }
      ],
      "edges" => [
        %{
          "source_entity_id" => "organization:google-worker-test-123",
          "target_entity_id" => "project:vertex-ai-worker-test-123",
          "edge_type" => "DEVELOPS",
          "confidence" => 0.9,
          "properties" => %{}
        },
        %{
          "source_entity_id" => "organization:google-worker-test-123",
          "target_entity_id" => "invalid:test",
          "edge_type" => "INVALID_EDGE"
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
    
    {:ok, _chunk} = Repo.insert(%Chunk{
      source_id: source.id,
      position: 1,
      text: "Google develops Vertex AI."
    })

    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => source.id, "mock" => true}})

    # Check entities
    entity = Repo.get_by(Entity, entity_id: "organization:google-worker-test-123", source_id: source.id)
    assert entity
    assert entity.name == "Google"
    assert entity.status == "pending_review"
    
    # Should skip invalid entity
    refute Repo.get_by(Entity, entity_id: "invalid:test", source_id: source.id)

    # Check edges
    edge = Repo.get_by(Edge, source_entity_id: "organization:google-worker-test-123", source_id: source.id)
    assert edge
    assert edge.target_entity_id == "project:vertex-ai-worker-test-123"
    assert edge.edge_type == "DEVELOPS"
    assert edge.status == "pending_review"

    # Should skip invalid edge
    refute Repo.get_by(Edge, edge_type: "INVALID_EDGE", source_id: source.id)

    # Check source status
    updated_source = Repo.get(Source, source.id)
    assert updated_source.status == "processing"
    assert updated_source.stage == "resolve"
  end
  
  test "worker handles no chunks correctly" do
    {:ok, source} = Repo.insert(%Source{
      uri: "https://example.com/no-chunks",
      raw_text: "No chunks here.",
      status: "pending_extraction",
      stage: "extract"
    })
    
    assert :ok == Worker.perform(%Oban.Job{args: %{"id" => source.id, "mock" => true}})
    
    updated_source = Repo.get(Source, source.id)
    assert updated_source.status == "skipped"
    assert updated_source.error == "No chunks to extract from"
  end
end
