defmodule AgentsKg.Repo do
  use Ecto.Repo,
    otp_app: :agents_kg,
    adapter: Ecto.Adapters.SQLite3
end
