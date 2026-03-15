defmodule AgentsKg.Source do
  use Ecto.Schema
  import Ecto.Changeset

  schema "sources" do
    field(:uri, :string)
    field(:title, :string)
    field(:type, :string, default: "url")
    field(:content_hash, :string)
    field(:raw_text, :string)
    field(:parsed_text, :string)
    field(:status, :string, default: "pending")
    field(:stage, :string, default: "fetch")
    field(:error, :string)
    field(:attempts, :integer, default: 0)
    field(:max_attempts, :integer, default: 5)

    timestamps(inserted_at: :created_at, type: :string)
  end

  def changeset(source, attrs) do
    source
    |> cast(attrs, [
      :uri,
      :title,
      :type,
      :content_hash,
      :raw_text,
      :parsed_text,
      :status,
      :stage,
      :error,
      :attempts,
      :max_attempts
    ])
    |> validate_required([:uri])
  end
end
