defmodule AgentsKg.Extractor.AgentTest do
  use ExUnit.Case, async: false

  alias AgentsKg.Extractor.Agent

  test "agent parses entities and edges correctly" do
    mock_response = %{
      "entities" => [
        %{
          "entity_id" => "organization:google",
          "name" => "Google",
          "type" => "Organization",
          "kind" => "company",
          "description" => "Search engine company",
          "aliases" => ["Alphabet"]
        }
      ],
      "edges" => [
        %{
          "source_entity_id" => "organization:google",
          "target_entity_id" => "project:vertex-ai",
          "edge_type" => "DEVELOPS",
          "confidence" => 0.9,
          "properties" => %{}
        }
      ]
    }

    ADK.LLM.Mock.set_responses([Jason.encode!(mock_response)])

    assert {:ok, result} = Agent.run("Google develops Vertex AI.", "test_session", model: "mock")

    assert length(result["entities"]) == 1
    entity = hd(result["entities"])
    assert entity["entity_id"] == "organization:google"
    assert entity["name"] == "Google"

    assert length(result["edges"]) == 1
    edge = hd(result["edges"])
    assert edge["source_entity_id"] == "organization:google"
    assert edge["edge_type"] == "DEVELOPS"
  end

  test "agent handles json decode error" do
    ADK.LLM.Mock.set_responses(["not valid json"])

    assert {:error, {:json_decode_error, _, "not valid json"}} =
             Agent.run("Some text", "test_session_2", model: "mock")
  end
end
