defmodule AgentsKg.Chunk do
  use Ecto.Schema
  import Ecto.Changeset

  schema "chunks" do
    belongs_to(:source, AgentsKg.Source, foreign_key: :source_id)
    field(:text, :string)
    field(:position, :integer)
    field(:section_heading, :string)
    field(:chunk_strategy, :string, default: "section")
    field(:token_count, :integer)
    field(:embedding, :binary)
    field(:embedding_model, :string)
    field(:embedded_at, :string)
  end

  def changeset(chunk, attrs) do
    chunk
    |> cast(attrs, [
      :source_id,
      :text,
      :position,
      :section_heading,
      :chunk_strategy,
      :token_count,
      :embedding,
      :embedding_model,
      :embedded_at
    ])
    |> validate_required([:source_id, :text, :position])
  end
end
