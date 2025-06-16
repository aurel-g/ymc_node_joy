# Standard library imports
# import os
# import toml

# Third-party imports
from yors_comfyui_node_setup import entry
from yors_pano_ansi_color import info_status, info_step, msg_padd, log_msg
from yors_pano_path_util import path_resolve,path_dirname,path_parse,path_comfy_get

__all__,NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS,NODE_MENU_NAMES = entry(__name__,__file__,False)

info_step(f"__all__ + WEB_DIRECTORY")
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# build(core): get root path with __file
root = path_resolve("../",__file__)

# build(core): get name from root path
name, stem, suffix, parent=path_parse(root)
comfy_root_path =  path_comfy_get(__file__,2,'./')


# comfy_root_path =  path_resolve('../../../',__file__)
# log_msg(f'comfy_root_path: {comfy_root_path}')

# comfy_model_path = path_comfy_get(__file__,2,'models')
# log_msg(f'comfy_model_path: {comfy_model_path}')

# comfy_model_path = path_comfy_get(__file__,'../../models')
# log_msg(f'comfy_model_path: {comfy_model_path}')

# comfy_model_path =  path_resolve('models',comfy_root_path)
# log_msg(f'comfy_model_path: {comfy_model_path}')


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
log_msg(f'comfy_root_path: {comfy_root_path}')
log_msg(msg_padd("=",60,"="))