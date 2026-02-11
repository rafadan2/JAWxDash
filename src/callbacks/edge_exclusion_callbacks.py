from dash import callback, dcc, Input, Output, State
import io
import logging
import os
import pandas as pd
import zipfile


# Local imports
from src.ellipsometry_toolbox.ellipsometry import Ellipsometry
from src.ellipsometry_toolbox.masking import create_masked_file
from src import ids
from src.utils.file_manager import get_file_path


logger = logging.getLogger(__name__)


@callback(
        Output(ids.Text.EXCLUDED_POINTS, "children"),
        Input(ids.DropDown.UPLOADED_FILES, "value"),
        Input(ids.Store.SETTINGS, "data"),
        State(ids.Store.UPLOADED_FILES, "data"),
)
def update_excluded_points_text(selected_file:str, settings:dict, stored_files:dict) -> str:

    # check if a file is selected and edge exclusion is turned on
    if not selected_file or not settings["ee_state"]:
        return ""
    
    # Loading into JAWFile object
    file_path = get_file_path(stored_files, selected_file)
    if not file_path:
        return ""

    file = Ellipsometry.from_path_or_stream(file_path)
    out_file = create_masked_file(file, settings)


    return "%i/%i" % (len(file.data.index) - len(out_file.data.index), len(file.data.index))



@callback(
    Output(ids.Download.EDGE_EXCLUDED_FILE, "data"),
    Input(ids.Button.DOWNLOAD_MASKED_DATA, "n_clicks"),
    State(ids.DropDown.UPLOADED_FILES, "value"),
    State(ids.Store.UPLOADED_FILES, "data"),
    State(ids.Store.SETTINGS, "data"),
)
def download_edge_exclusion(n_clicks, selected_file:str, stored_files:dict, settings:dict):
    

    # check if a file is selected and edge exclusion is turned on
    if not selected_file or not settings["ee_state"]:
        return None

    
    if settings["batch_processing"]:
        """
        Batch processing is selected and all the files in the 'file_manager' will be processed
        and downloaded as a zip-file
        """

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

            file_names = []
            n_points = []
            thickness_avg = []
            thickness_std = []
            mse_avg = []
            mse_std = []
            sigint_avg = []
            sigint_std = []
            for selected_file in stored_files:
                # File output name
                root, ext = os.path.splitext(selected_file)
                file_name = root + "_masked" + ext


                # Loading into JAWFile object
                file_path = get_file_path(stored_files, selected_file)
                if not file_path:
                    continue

                file = Ellipsometry.from_path_or_stream(file_path)
                masked_file = create_masked_file(file, settings)

                file_names.append(file_name)
                n_points.append(len(masked_file.data.index))
                thickness_avg.append(masked_file.data["Thickness # 1 (nm)"].mean())
                thickness_std.append(masked_file.data["Thickness # 1 (nm)"].std())
                mse_avg.append(masked_file.data["MSE"].mean())
                mse_std.append(masked_file.data["MSE"].std())
                sigint_avg.append(masked_file.data["SigInt"].mean())
                sigint_std.append(masked_file.data["SigInt"].std())


                buffer = masked_file.to_buffer()
                zf.writestr(file_name, buffer.getvalue())
            
            
            stats = pd.DataFrame(data={
                "File Name": file_names,
                "# Points": n_points,
                "Avg. Thickness": thickness_avg,
                "Std. Thickness": thickness_std,
                "Avg. MSE": mse_avg,
                "Std. MSE": mse_std,
                "Avg. SigInt": sigint_avg,
                "Std. SigInt": sigint_std
            })

            buffer = io.StringIO()
            stats.to_csv(buffer, sep="\t", float_format="%.4f", header=True, index=False)
            zf.writestr("ellipsometer_summary.txt", buffer.getvalue())

        zip_buffer.seek(0)
        

        return dcc.send_bytes(zip_buffer.getvalue(), filename="ellipsometer_download.zip")           
        

    else:
        """
        Single file processing
        """
        

        # File output name
        root, ext = os.path.splitext(selected_file)
        file_name = os.path.join(root, "_masked", ext)
    

        # Loading into JAWFile object
        file_path = get_file_path(stored_files, selected_file)
        if not file_path:
            return None

        file = Ellipsometry.from_path_or_stream(file_path)
        masked_file = create_masked_file(file, settings)
        
        buffer = masked_file.to_buffer()
        
        return dcc.send_string(buffer.getvalue(), filename=file_name)
