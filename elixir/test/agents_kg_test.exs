defmodule AgentsKgTest do
  use ExUnit.Case
  doctest AgentsKg

  test "greets the world" do
    assert AgentsKg.hello() == :world
  end
end
