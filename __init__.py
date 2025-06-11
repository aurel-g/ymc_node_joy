# Standard library imports
from pathlib import Path
# import os
# import toml

# Third-party imports
from yors_comfyui_node_setup import entry
from yors_pano_ansi_color import info_status, info_step, msg_padd, log_msg

# def get_version_from_pyproject(file: str, fallback: str = '1.0.0'):
#     """
#     get version from pyproject.toml 's project.version
#     """
#     try:
#         file = Path(file)
#         if not file.exists():
#             raise FileNotFoundError
#         if not os.access(file, os.R_OK):
#             raise PermissionError(f"Permission denied: {file}")
            
#         pyproject_content = file.read_text()
#         pyproject = toml.loads(pyproject_content)
#         version = pyproject.get('project', {}).get('version', fallback)
#     except FileNotFoundError:
#         log_msg(f"pyproject.toml not found at {file}, using default version {fallback}")
#         version = fallback
#     except PermissionError as e:
#         log_msg(f"Permission error: {e}, using default version {fallback}")
#         version = fallback
#     except Exception as e:
#         log_msg(f"Error reading pyproject.toml: {e}, using default version {fallback}")
#         version = fallback
#     return version

# def get_version_from_txt(file: str, fallback: str = '1.0.0'):
#     """
#     get version from version.txt file
#     """
#     try:
#         file = Path(file)
#         if not file.exists():
#             raise FileNotFoundError
#         if not os.access(file, os.R_OK):
#             raise PermissionError(f"Permission denied: {file}")
            
#         version = file.read_text().strip()
#         if not version:
#             raise ValueError("Empty version string")
#         return version
#     except Exception as e:
#         log_msg(f"Error reading version.txt: {e}, using default version {fallback}")
#         return fallback

def path_resolve(path: str,root:str):
    """
    resolve path to absolute path with root path

    INIT_PY_REl='../'
    root = path_resolve(INIT_PY_REl,str(Path(__file__)))
    path_resolve(root,'pyproject.toml')
    """
    return str(Path(root).joinpath(path).resolve().as_posix())

def path_dirname(path: str):
    """
    get dirname of path
    """
    return str(Path(path).parent)

def path_parse(path:str):
    flag = Path(path)
    name = flag.name
    stem = flag.stem
    suffix = flag.suffix
    parent = str(flag.parent.as_posix())
    return (name, stem, suffix, parent)


__all__,NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS,NODE_MENU_NAMES = entry(__name__,__file__,False)

info_step(f"__all__ + WEB_DIRECTORY")
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# build(core): get root path with __file and INIT_PY_REl
INIT_PY_REl="../"
root = path_resolve(INIT_PY_REl,str(Path(__file__)))

# build(core): get name from root path
name, stem, suffix, parent=path_parse(root)


# - build(core): get version from pyproject.toml
# pyproject_file=path_resolve(root,'pyproject.toml')
# version= get_version_from_pyproject(Path(pyproject_file))
# version_file=path_resolve(root,'version.txt')
# version=get_version_from_txt(version_file)

# name='ymc_node_joy'
# version='1.0.0'
log_msg(msg_padd("=",60,"=")) 
log_msg(msg_padd(f'welocme to {name}',60,"="))
# log_msg(f'version: {version}')
log_msg(f'node counts:{len(NODE_MENU_NAMES)}')
log_msg(f'node menu names:')
NODE_MENU_NAMES.sort()
for node_name in NODE_MENU_NAMES:
    # log_msg(f'node name:{node_name}')
    info_status(f'{node_name}',0)
log_msg(msg_padd("=",60,"="))