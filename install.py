from transformers import pipeline, SamModel, SamProcessor
import os

def main():    
  # set HF_HOME env var
  comfy_path = os.environ.get('COMFYUI_PATH')
  if comfy_path is None:
      comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
  
  model_path = os.path.abspath(os.path.join(comfy_path, 'models'))
  owl_path = os.path.join(model_path, 'owlsam')
  os.environ["HF_HOME"] = owl_path
  if not os.path.exists(owl_path):
    os.makedirs(owl_path)

  checkpoint = "google/owlv2-base-patch16-ensemble"
  detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")
  detector.save_pretrained(owl_path)

  SamModel.from_pretrained("facebook/sam-vit-base")
  SamProcessor.from_pretrained("facebook/sam-vit-base")

if __name__ == "__main__":
    main()
