import Config

config :agents_kg, AgentsKg.Repo,
  adapter: Ecto.Adapters.SQLite3,
  database: "../pipeline.db",
  pool_size: 5

config :agents_kg, ecto_repos: [AgentsKg.Repo]

if File.exists?(Path.expand("#{config_env()}.exs", __DIR__)) do
  import_config "#{config_env()}.exs"
end

config :agents_kg, Oban,
  engine: Oban.Engines.Lite,
  repo: AgentsKg.Repo,
  queues: [default: 10]
