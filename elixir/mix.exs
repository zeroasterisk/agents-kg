defmodule AgentsKg.MixProject do
  use Mix.Project

  def project do
    [
      app: :agents_kg,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {AgentsKg.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ecto_sql, "~> 3.11"},
      {:ecto_sqlite3, "~> 0.16"},
      {:jason, "~> 1.4"},
      {:oban, "~> 2.18"},
      {:req, "~> 0.5.0"},
      {:adk, path: "../../adk-elixir"},
      {:phoenix, "~> 1.7"},
      {:phoenix_html, "~> 4.0"},
      {:phoenix_live_view, "~> 1.0"},
      {:lazy_html, ">= 0.1.0", only: :test},
      {:bandit, "~> 1.0"},
      {:floki, "~> 0.36.0"},
      {:a2a, "~> 0.2.0"},
      {:opentelemetry_api, "~> 1.0"}
    ]
  end
end
