defmodule AgentsKgWeb.AuditLive do
  use AgentsKgWeb, :live_view
  alias AgentsKg.{Repo, Entity}
  import Ecto.Query

  def mount(_params, _session, socket) do
    if connected?(socket) do
      # Subscribe if needed
    end

    socket =
      assign(socket,
        entities: load_needs_human(),
        selected_entity: nil,
        editing: false
      )

    {:ok, socket}
  end

  defp load_needs_human do
    Entity
    |> where(status: "needs_human")
    |> order_by(desc: :id)
    |> Repo.all()
  end

  def render(assigns) do
    ~H"""
    <div class="flex gap-8">
      <div class="w-1/3 border-r pr-4">
        <h2 class="text-xl font-bold mb-4">Needs Human Review</h2>
        <ul class="space-y-2">
          <%= for entity <- @entities do %>
            <li class={"p-4 border rounded shadow-sm cursor-pointer hover:bg-gray-50 flex justify-between " <> if @selected_entity && @selected_entity.id == entity.id, do: "bg-blue-50 border-blue-200", else: "bg-white"} phx-click="select_entity" phx-value-id={entity.id}>
              <div>
                <div class="font-bold"><%= entity.name %></div>
                <div class="text-xs text-gray-500"><%= entity.type %></div>
              </div>
              <div>
                <button class="text-blue-500 hover:text-blue-700 font-semibold" phx-click="approve" phx-value-id={entity.id}>Approve</button>
              </div>
            </li>
          <% end %>
          <%= if @entities == [] do %>
            <div class="text-gray-500 italic">No entities need review.</div>
          <% end %>
        </ul>
      </div>

      <div class="w-2/3 pl-4">
        <%= if @selected_entity do %>
          <div class="bg-white p-6 rounded shadow">
            <h2 class="text-2xl font-bold mb-4">Review: <%= @selected_entity.name %></h2>
            
            <%= if @editing do %>
              <form phx-submit="save_edit" class="space-y-4">
                <div>
                  <label class="block font-bold">Name</label>
                  <input type="text" name="name" value={@selected_entity.name} class="border p-2 w-full rounded" />
                </div>
                <div>
                  <label class="block font-bold">Type</label>
                  <input type="text" name="type" value={@selected_entity.type} class="border p-2 w-full rounded" />
                </div>
                <div>
                  <label class="block font-bold">Kind</label>
                  <input type="text" name="kind" value={@selected_entity.kind} class="border p-2 w-full rounded" />
                </div>
                <div>
                  <label class="block font-bold">Description</label>
                  <textarea name="description" class="border p-2 w-full rounded" rows="3"><%= @selected_entity.description %></textarea>
                </div>
                <div>
                  <label class="block font-bold">Aliases (JSON string)</label>
                  <input type="text" name="aliases" value={@selected_entity.aliases} class="border p-2 w-full rounded" />
                </div>
                
                <div class="flex gap-4 pt-4 border-t">
                  <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded shadow hover:bg-blue-700">Save</button>
                  <button type="button" phx-click="cancel_edit" class="bg-gray-200 px-4 py-2 rounded hover:bg-gray-300">Cancel</button>
                </div>
              </form>
            <% else %>
              <div class="mb-4">
                <strong>Type:</strong> <%= @selected_entity.type %>
              </div>
              <div class="mb-4">
                <strong>Kind:</strong> <%= @selected_entity.kind || "N/A" %>
              </div>
              <div class="mb-4">
                <strong>Description:</strong> <%= @selected_entity.description %>
              </div>
              <div class="mb-4">
                <strong>Aliases:</strong> <%= @selected_entity.aliases %>
              </div>

              <div class="mt-6 border-t pt-4">
                <h3 class="text-lg font-semibold mb-2">Actions</h3>
                <div class="flex gap-4">
                  <button class="bg-green-600 text-white px-4 py-2 rounded shadow hover:bg-green-700" phx-click="approve" phx-value-id={@selected_entity.id}>Approve</button>
                  <button class="bg-yellow-500 text-white px-4 py-2 rounded shadow hover:bg-yellow-600" phx-click="edit">Edit</button>
                  <button class="bg-red-600 text-white px-4 py-2 rounded shadow hover:bg-red-700" phx-click="reject" phx-value-id={@selected_entity.id}>Reject</button>
                </div>
              </div>
              
              <div class="mt-8 border-t pt-4">
                <h3 class="text-lg font-semibold mb-2">Merge into existing</h3>
                <form phx-submit="merge">
                  <input type="hidden" name="source_id" value={@selected_entity.id} />
                  <div class="flex gap-2">
                    <input type="text" name="target_id" placeholder="Target Entity ID..." class="border rounded p-2 flex-1" />
                    <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded shadow hover:bg-blue-700">Merge</button>
                  </div>
                </form>
              </div>
            <% end %>
          </div>
        <% else %>
          <div class="text-gray-500 text-center mt-12">Select an entity from the list to review.</div>
        <% end %>
      </div>
    </div>
    """
  end

  def handle_event("select_entity", %{"id" => id}, socket) do
    id_int = String.to_integer(id)
    entity = Enum.find(socket.assigns.entities, &(&1.id == id_int))
    {:noreply, assign(socket, selected_entity: entity, editing: false)}
  end

  def handle_event("edit", _, socket) do
    {:noreply, assign(socket, editing: true)}
  end

  def handle_event("cancel_edit", _, socket) do
    {:noreply, assign(socket, editing: false)}
  end

  def handle_event("save_edit", params, socket) do
    entity = Repo.get!(Entity, socket.assigns.selected_entity.id)

    attrs = %{
      name: params["name"],
      type: params["type"],
      kind: params["kind"],
      description: params["description"],
      aliases: params["aliases"]
    }

    entity = Ecto.Changeset.change(entity, attrs) |> Repo.update!()

    socket =
      assign(socket,
        entities: load_needs_human(),
        selected_entity: entity,
        editing: false
      )

    {:noreply, socket}
  end

  def handle_event("approve", %{"id" => id}, socket) do
    entity = Repo.get!(Entity, String.to_integer(id))

    Ecto.Changeset.change(entity, status: "approved")
    |> Repo.update!()

    socket = assign(socket, entities: load_needs_human(), selected_entity: nil, editing: false)
    {:noreply, socket}
  end

  def handle_event("reject", %{"id" => id}, socket) do
    entity = Repo.get!(Entity, String.to_integer(id))

    Ecto.Changeset.change(entity, status: "rejected")
    |> Repo.update!()

    socket = assign(socket, entities: load_needs_human(), selected_entity: nil, editing: false)
    {:noreply, socket}
  end

  def handle_event("merge", %{"source_id" => source_id, "target_id" => target_id}, socket) do
    if target_id != "" do
      entity = Repo.get!(Entity, String.to_integer(source_id))

      Ecto.Changeset.change(entity, status: "merged", merged_into: target_id)
      |> Repo.update!()

      socket = assign(socket, entities: load_needs_human(), selected_entity: nil, editing: false)
      {:noreply, socket}
    else
      {:noreply, socket}
    end
  end
end
