defmodule AgentsKg.Entity do
  use Ecto.Schema
  import Ecto.Changeset

  schema "entities" do
    field(:entity_id, :string)
    field(:name, :string)
    field(:kind, :string)
    field(:type, :string)
    field(:description, :string)
    field(:aliases, :string, default: "[]")
    field(:status, :string, default: "pending_review")
    field(:merged_into, :string)

    belongs_to(:source, AgentsKg.Source, foreign_key: :source_id)
    field(:chunk_id, :integer)

    timestamps(inserted_at: :created_at, type: :utc_datetime)
  end

  def changeset(entity, attrs) do
    entity
    |> cast(attrs, [
      :entity_id,
      :name,
      :kind,
      :type,
      :description,
      :aliases,
      :status,
      :merged_into,
      :source_id,
      :chunk_id
    ])
    |> validate_required([:entity_id, :name, :type])
  end
end
