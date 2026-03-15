defmodule AgentsKg.Edge do
  use Ecto.Schema
  import Ecto.Changeset

  schema "edges" do
    field(:edge_id, :string)
    field(:source_entity_id, :string)
    field(:target_entity_id, :string)
    field(:edge_type, :string)
    field(:properties, :string, default: "{}")
    field(:valid_from, :string)
    field(:valid_to, :string)
    field(:confidence, :float, default: 0.5)
    field(:chunk_id, :integer)
    belongs_to(:source, AgentsKg.Source, foreign_key: :source_id)
    field(:extracted_at, :string)
    field(:source_type, :string, default: "automated")
    field(:status, :string, default: "pending_review")

    timestamps(inserted_at: :created_at, type: :utc_datetime)
  end

  def changeset(edge, attrs) do
    edge
    |> cast(attrs, [
      :edge_id,
      :source_entity_id,
      :target_entity_id,
      :edge_type,
      :properties,
      :valid_from,
      :valid_to,
      :confidence,
      :chunk_id,
      :source_id,
      :extracted_at,
      :source_type,
      :status
    ])
    |> validate_required([:edge_id, :source_entity_id, :target_entity_id, :edge_type])
  end
end
