import Config

config :agents_kg, AgentsKg.Repo,
  adapter: Ecto.Adapters.SQLite3,
  database: "../pipeline.db",
  pool: Ecto.Adapters.SQL.Sandbox

config :agents_kg, Oban, testing: :manual

config :agents_kg, AgentsKgWeb.Endpoint, server: false

config :agents_kg, AgentsKgWeb.Endpoint,
  secret_key_base: "a_very_long_secret_key_base_that_is_at_least_sixty_four_bytes_long_ok"
