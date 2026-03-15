defmodule AgentsKg.RepoTest do
  use ExUnit.Case
  alias AgentsKg.Repo
  alias AgentsKg.Entity

  alias AgentsKg.Source
  alias AgentsKg.Edge

  import Ecto.Query

  test "can query the existing database" do
    result = Repo.all(from(Entity, limit: 1))
    assert is_list(result)

    result_sources = Repo.all(from(Source, limit: 1))
    assert is_list(result_sources)

    result_edges = Repo.all(from(Edge, limit: 1))
    assert is_list(result_edges)
  end
end
