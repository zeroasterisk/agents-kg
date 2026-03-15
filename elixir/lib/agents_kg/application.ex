defmodule AgentsKg.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      AgentsKg.Repo,
      {Phoenix.PubSub, name: AgentsKg.PubSub},
      {Oban, Application.fetch_env!(:agents_kg, Oban)},
      # AgentsKgWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: AgentsKg.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
