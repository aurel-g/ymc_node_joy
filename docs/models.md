
## Required Models

Essential components:
- `google/siglip-so400m-patch14-384` (Vision model)
- `unsloth/Meta-Llama-3.1-8B-bnb-4bit` or `meta-llama/Meta-Llama-3.1-8B` (LLM)
- `Joy_caption/image_adapter.pt` (Custom adapter)

```plaintext
<comfyui_root>/
├── models/
│   ├── clip/                    # SigLIP Vision Model
│   │   └── siglip-so400m-patch14-384/
│   ├── llm/                     # Llama Language Model
│   │   ├── Meta-Llama-3.1-8B-bnb-4bit/
│   │   └── Meta-Llama-3.1-8B/
│   └── Joy_caption/             # Custom Components
│       └── image_adapter.pt     # Dimension Adapter
```

### 1. SigLIP Vision Model (google/siglip-so400m-patch14-384)
**International**: https://huggingface.co/google/siglip-so400m-patch14-384  
**China Mirror**: https://hf-mirror.com/google/siglip-so400m-patch14-384

### 2. Llama Language Models
#### 4bit Quantized (unsloth/Meta-Llama-3.1-8B-bnb-4bit)
**International**: https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit  
**China Mirror**: https://hf-mirror.com/unsloth/Meta-Llama-3.1-8B-bnb-4bit

#### Original (meta-llama/Meta-Llama-3.1-8B)
**International**: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B (Access approval required)  
**China Mirror**: https://hf-mirror.com/meta-llama/Meta-Llama-3.1-8B

### 3. Image Adapter (Joy_caption/image_adapter.pt)
**International**: https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6  
**China Mirror**: https://www.modelscope.cn/models/fireicewolf/joy-caption-pre-alpha/files

