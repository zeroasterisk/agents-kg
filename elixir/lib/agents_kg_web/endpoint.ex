defmodule AgentsKgWeb.Endpoint do
  use Phoenix.Endpoint, otp_app: :agents_kg

  plug Plug.Static,
    at: "/",
    from: :agents_kg,
    gzip: false,
    only: AgentsKgWeb.static_paths()

  plug Plug.RequestId
  plug Plug.Telemetry, event_prefix: [:phoenix, :endpoint]

  plug Plug.Parsers,
    parsers: [:urlencoded, :multipart, :json],
    pass: ["*/*"],
    json_decoder: Phoenix.json_library()

  plug Plug.MethodOverride
  plug Plug.Head
  plug Plug.Session, store: :cookie, key: "_agents_kg_key", signing_salt: "q4B6Qz+O"
  plug AgentsKgWeb.Router
end

