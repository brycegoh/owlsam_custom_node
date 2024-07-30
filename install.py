import folder_paths
from transformers import pipeline, SamModel, SamProcessor
import os

def main():    
  # using python os ensure folder exists if not create
  owl_path = folder_paths.get_full_path("owlsam", "owlv2")
  sam_path = folder_paths.get_full_path("owlsam", "sam")

  if not os.path.exists(owl_path):
    os.makedirs(owl_path)
  if not os.path.exists(sam_path):
    os.makedirs(sam_path)

  checkpoint = "google/owlv2-base-patch16-ensemble"
  detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device="cuda")
  detector.save_pretrained(owl_path)

  sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
  sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

  sam_model.save_pretrained(folder_paths.get_full_path("owlsam", "sam"))
  sam_processor.save_pretrained(folder_paths.get_full_path("owlsam", "sam"))
  

if __name__ == "__main__":
    main()