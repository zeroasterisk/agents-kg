defmodule AgentsKgWeb.AuditLiveTest do
  use ExUnit.Case, async: true
  import Phoenix.ConnTest
  import Phoenix.LiveViewTest
  alias AgentsKg.{Repo, Entity}

  @endpoint AgentsKgWeb.Endpoint

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(AgentsKg.Repo)
    %{conn: Phoenix.ConnTest.build_conn()}
  end

  test "disconnected and connected render", %{conn: conn} do
    {:ok, page_live, disconnected_html} = live(conn, "/")
    assert disconnected_html =~ "Needs Human Review"
    assert render(page_live) =~ "Needs Human Review"
  end

  test "can select and approve entity", %{conn: conn} do
    entity =
      Repo.insert!(%Entity{
        entity_id: "test:ent1",
        name: "Test Entity 1",
        type: "Project",
        status: "needs_human"
      })

    {:ok, page_live, _html} = live(conn, "/")
    assert render(page_live) =~ "Test Entity 1"

    # Select
    page_live
    |> element("li[phx-click=\"select_entity\"][phx-value-id=\"#{entity.id}\"]")
    |> render_click()

    assert render(page_live) =~ "Review: Test Entity 1"

    # Approve (using the green button in detail view)
    page_live
    |> element("button.bg-green-600", "Approve")
    |> render_click(%{"id" => to_string(entity.id)})

    # Entity is approved, shouldn't be in the list
    refute render(page_live) =~ "Review: Test Entity 1"

    assert Repo.get!(Entity, entity.id).status == "approved"
  end

  test "can edit and save entity", %{conn: conn} do
    entity =
      Repo.insert!(%Entity{
        entity_id: "test:ent2",
        name: "Test Entity 2",
        type: "Person",
        status: "needs_human"
      })

    {:ok, page_live, _html} = live(conn, "/")

    # Select
    page_live
    |> element("li[phx-click=\"select_entity\"][phx-value-id=\"#{entity.id}\"]")
    |> render_click()

    # Click edit
    page_live
    |> element("button.bg-yellow-500", "Edit")
    |> render_click()

    assert render(page_live) =~ "Cancel"

    # Save
    page_live
    |> form("form[phx-submit=\"save_edit\"]", %{
      "name" => "Updated Name",
      "type" => "Project",
      "kind" => "library",
      "description" => "New description",
      "aliases" => "[]"
    })
    |> render_submit()

    # Ensure details view reflects it
    assert render(page_live) =~ "Updated Name"
    assert render(page_live) =~ "library"

    # Check db
    updated = Repo.get!(Entity, entity.id)
    assert updated.name == "Updated Name"
    assert updated.type == "Project"
    assert updated.kind == "library"
  end
end
