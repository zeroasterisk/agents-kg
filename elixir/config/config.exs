import Config

config :agents_kg, AgentsKg.Repo,
  adapter: Ecto.Adapters.SQLite3,
  database: "../pipeline.db",
  pool_size: 5

config :agents_kg, ecto_repos: [AgentsKg.Repo]

config :agents_kg, Oban,
  engine: Oban.Engines.Lite,
  repo: AgentsKg.Repo,
  queues: [default: 10]

if File.exists?(Path.expand("#{config_env()}.exs", __DIR__)) do
  import_config "#{config_env()}.exs"
end

config :agents_kg, AgentsKgWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [html: AgentsKgWeb.ErrorHTML],
    layout: false
  ],
  pubsub_server: AgentsKg.PubSub,
  live_view: [signing_salt: "q4B6Qz+O"]

config :phoenix, :json_library, Jason

