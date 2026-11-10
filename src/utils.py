import time
import os
import inspect
import random
import string
import shutil


def format_time(seconds):
    '''
    seconds: int
    '''
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    ret = ""
    if hours > 0:
        ret += f"{hours}h "
    if minutes > 0 or hours > 0:
        ret += f"{minutes}m "
    ret += f"{seconds}s"
    return ret

def format_number(n):
    '''
    return number in ?K or ?M if it is millions of thousands
    '''
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)

def clear_line():
    '''
    clear the current line in the console
    '''
    print("\r" + " " * 150 + "\r", end='\r')

def format_header(txt, line="="):
    return f"\n{line*len(txt)}\n{txt}\n{line*len(txt)}"

def format_separator(txt, line="-"):
    return f"\n{line*len(txt)}\n{txt}"

# Custom Print function
def cp(color, text, verbose=0, CONFIG=None, end="\n", send_cloud=False):

    if CONFIG is None:
        CONFIG = {}
    # set defaults if not defined
    if 'verbose' not in CONFIG:
        CONFIG['verbose'] = 2
    if verbose is None:
        verbose = CONFIG['verbose']

    # Define a mapping of color names to ANSI escape codes
    color_map = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'gray': '\033[90m',
        'orange': '\033[38;5;208m'  # Extended 256-color for orange
    }

    reset = '\033[0m'  # Reset color
    
    timestamphmse = time.strftime("%H:%M:%S", time.localtime())

    if CONFIG.get('verbose', 0) >= verbose:
        color_code = color_map.get(color.lower(), '')  # Default to no color if invalid
        # timestamp in format: [hhhh:mm:ss]. Time is now
        print(f"{color_map['gray']}[{timestamphmse}] {color_code}{text}{reset}", end=end)


def get_abs_path(relative_path):
    """
    Get absolute path based on the relative path of the script that calls this function
    """
    # Get the filename of the caller
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename

    base_path = os.path.dirname(os.path.abspath(caller_file))
    abs_path = os.path.join(base_path, relative_path)

    path = os.path.abspath(abs_path)

    return path

def get_tmp_path(relative_path=""):
    return get_abs_path(f"../tmp/{relative_path}")

def empty_dir(dir_path):
    """
    Empties the specified directory by deleting all files and subdirectories within it.
    If the directory does not exist, it is created.

    dir_path: Path to the directory to be emptied.
    """
    if not os.path.exists(dir_path):
        cp("yellow", f"Directory does not exist. Creating: {dir_path}")
        os.makedirs(dir_path)
        return

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or link
                cp("gray", f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
                cp("gray", f"Deleted directory: {file_path}")
        except Exception as e:
            cp("red", f"Failed to delete {file_path}. Reason: {e}")

def random_string(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


def ask_yes_no(prompt):
    while True:
        ans = input(f"{prompt} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False

