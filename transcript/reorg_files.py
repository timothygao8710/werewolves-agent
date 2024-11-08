import os
import shutil
import json
import os
import shutil

def reorg_files(folder, game_log_file):
    folder = folder.strip()
    # Get all jsonl files in the specified folder
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and (f.endswith('.jsonl') or f.endswith('.txt'))]

    for file_name in files:
        # Split the file name into parts
        parts = file_name.split('_')
        if len(parts) == 4:
            player_name, gameid1, gameid2, type_of_file = parts
            # Create the game directory if it doesn't exist
            game_dir = os.path.join(folder, f"GAME_ID_{gameid1}_{gameid2}")
            os.makedirs(game_dir, exist_ok=True)
            # Move the file to the game directory
            shutil.move(os.path.join(folder, file_name), os.path.join(game_dir, f"{player_name}_{type_of_file}"))
        else:
            # Create the error_logs directory if it doesn't exist
            error_logs_dir = os.path.join(folder, 'error_logs')
            os.makedirs(error_logs_dir, exist_ok=True)
            # Move the file to the error_logs directory
            shutil.move(os.path.join(folder, file_name), os.path.join(error_logs_dir, file_name))
    
    shutil.move(game_log_file, os.path.join(game_dir, game_log_file))