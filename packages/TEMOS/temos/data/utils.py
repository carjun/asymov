from pathlib import Path
import pdb

def get_split_keyids(path: str, split: str):
    # pdb.set_trace()
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")
