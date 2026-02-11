# Package import
from dash import html, callback, Output, Input, State, ctx, no_update

# Local import
from src import ids
from src.utils.file_handler import save_upload
from src.utils.file_manager import get_file_title


@callback(
    Output(ids.Store.UPLOADED_FILES, 'data', allow_duplicate=True),
    Output(ids.Div.INFO, 'children', allow_duplicate=True),
    Input(ids.Upload.DRAG_N_DROP, 'contents'),
    State(ids.Upload.DRAG_N_DROP, 'filename'),
    State(ids.Store.UPLOADED_FILES, 'data'),
    prevent_initial_call=True,
)
def update_uploaded_files(contents, filenames:str, stored_files:dict):
    
    # check if the 'contents' is NOT none
    if not contents:
        return stored_files or {}, no_update

    stored_files = stored_files or {}
        
    # iterate over the contents and filename pairs
    for content, filename in zip(contents, filenames):

        # check if the file is already loaded
        if filename not in stored_files:
            path = save_upload(content, filename)

            stored_files[filename] = {
                "path": path,
                "title": filename,
            }
        else:
            entry = stored_files[filename]
            if not isinstance(entry, dict):
                stored_files[filename] = {
                    "path": entry,
                    "title": filename,
                }
            elif not entry.get("title"):
                entry["title"] = filename
                stored_files[filename] = entry
    

    return stored_files, html.Div("Uploaded: " + ', '.join([name for name in filenames]))


@callback(
    Output(ids.Input.FILE_TITLE, "value"),
    Input(ids.DropDown.UPLOADED_FILES, "value"),
    Input(ids.Store.UPLOADED_FILES, "data"),
)
def update_file_title_input(selected_file, stored_files):
    title = get_file_title(stored_files, selected_file)
    return title or ""


@callback(
    Output(ids.Store.UPLOADED_FILES, "data", allow_duplicate=True),
    Input(ids.Input.FILE_TITLE, "value"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    prevent_initial_call=True,
)
def update_file_title(title, selected_file, stored_files):
    if not selected_file or not stored_files or selected_file not in stored_files:
        return no_update

    normalized_title = (title or "").strip()
    if not normalized_title:
        normalized_title = selected_file

    entry = stored_files[selected_file]
    if isinstance(entry, dict):
        if entry.get("title") == normalized_title:
            return no_update
        entry["title"] = normalized_title
        stored_files[selected_file] = entry
    else:
        stored_files[selected_file] = {
            "path": entry,
            "title": normalized_title,
        }

    return stored_files



@callback(
    Output(ids.Store.UPLOADED_FILES, 'data'),
    Output(ids.DropDown.UPLOADED_FILES, 'options'),
    Output(ids.Div.INFO, 'children'),
    Input(ids.Button.DELETE_SELECTED, 'n_clicks'),
    Input(ids.Button.CLEAR_FILE_MANAGER, "n_clicks"),
    State(ids.DropDown.UPLOADED_FILES, 'value'),
    State(ids.Store.UPLOADED_FILES, 'data'),
    prevent_initial_call=True,
)
def delete_selected_from_list(delete, clear, selected_file, current_files):

    triggered_id = ctx.triggered_id  # This tells you which button was clicked

    # Removing selected file from dropdown
    if triggered_id == ids.Button.DELETE_SELECTED:

        new_options = [f for f in current_files if f != selected_file]
        del current_files[selected_file]

        return current_files, new_options, html.Div(f"Deleted: {selected_file}")
    

    elif triggered_id == ids.Button.CLEAR_FILE_MANAGER:
        return {}, [], html.Div("File manager was cleared")
