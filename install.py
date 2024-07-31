from transformers import AutoProcessor, Owlv2ForObjectDetection, SamModel, SamProcessor
import os

def main():    
  # set HF_HOME env var
  comfy_path = os.environ.get('COMFYUI_PATH')
  if comfy_path is None:
      comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
  
  model_path = os.path.abspath(os.path.join(comfy_path, 'models'))
  os.environ["HF_HOME"] = model_path
  if not os.path.exists(model_path):
    os.makedirs(model_path)

  SamModel.from_pretrained("facebook/sam-vit-base")
  SamProcessor.from_pretrained("facebook/sam-vit-base")
  processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
  model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

if __name__ == "__main__":
    main()
