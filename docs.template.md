<div align="center">
  <h1>{node_name}</h1>
  <p>
    <strong>🤖 {node_desc} </strong>
  </p>

</div>

<!-- inject desc here -->
<!-- inject-desc -->

## Why

<!-- inject why here -->
<!-- inject-why -->


## Features

<!-- inject feat here -->
<!-- inject-features -->

## Nodes

<!-- inject node here -->
<!-- inject-nodes -->

## Install 

```bash
# cd to comfyui/custom_nodes
git clone https://github.com/ymc-github/{node_name}
```
- **deps will be installed automatically** if deps in requirements.txt were not installed when comfyui up

<!-- inject model here -->
<!-- inject-models -->


## Usage

- you can find it in search box : double click + typing keywords (eg: `joy - `, `caption`)
- you can find it in right mouse menu : `{node_right_menu}`
- ~~you can find it in right mouse menu : `utils/{node_right_menu}` (as alias)~~

## Based-on

- pypi package [yors_comfyui_node_setup](https://pypi.org/project/yors_comfyui_node_setup/) -  setup comfyui custom nodes easily
- pypi package [yors_pano_ansi_color](https://pypi.org/project/yors_pano_ansi_color/) - info msg in console with color in your comfyui custom nodes easily
- ~~pypi package [yors_pano_zero_field](https://pypi.org/project/yors_pano_zero_field/) - set nodes input field to be HQ in your comfyui custom nodes easily~~


## Published to Comfy registry

- get more details in [publish_to_comfy.yml](.github/workflows/publish_to_comfy.yml)

- [docs for publishing to comfy registey](https://docs.comfy.org/registry/overview)

- installed with comfy-cli ? `comfy node registry-install {node_name}`

## Author

ymc-github <ymc.github@gmail.com>

name|email|desciption
:--|:--|:--
yemiancheng|<ymc.github@gmail.com>|Main developer and code maintainer|
chenxinghua|<455758525@qq.com>|Code reference from [StartHua/Comfyui_CXH_joy_caption](https://github.com/StartHua/Comfyui_CXH_joy_caption)|

## License

MIT