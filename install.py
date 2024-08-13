from transformers import AutoProcessor, Owlv2ForObjectDetection, SamModel, SamProcessor
import os



def main():    
  os.system("pip install huggingface_hub")
  os.system("git clone https://github.com/facebookresearch/segment-anything-2.git")
  os.system('cd segment-anything-2 & pip install -e ".[demo]" & python setup.py install')

  from huggingface_hub import hf_hub_download
  # set HF_HOME env var
  comfy_path = os.environ.get('COMFYUI_PATH')
  if comfy_path is None:
      comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
  
  model_path = os.path.abspath(os.path.join(comfy_path, 'models'))

  sam2_path = os.path.abspath(os.path.join(model_path, 'sam2'))
  hf_hub_download(repo_id = "facebook/sam2-hiera-large", filename="sam2_hiera_large.pt", local_dir =sam2_path)
  hf_hub_download(repo_id = "facebook/sam2-hiera-large", filename="sam2_hiera_l.yaml", local_dir =sam2_path)


  os.environ["HF_HOME"] = model_path
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  print(f"Set HF_HOME to {model_path}")
  SamModel.from_pretrained("facebook/sam-vit-base", cache_dir=model_path)
  SamProcessor.from_pretrained("facebook/sam-vit-base", cache_dir=model_path)
  AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir=model_path)
  Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir=model_path)

if __name__ == "__main__":
    main()
