import Config

config :agents_kg, AgentsKg.Repo,
  adapter: Ecto.Adapters.SQLite3,
  database: "../pipeline.db",
  pool: Ecto.Adapters.SQL.Sandbox
