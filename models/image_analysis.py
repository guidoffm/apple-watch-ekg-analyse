from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import torch
import os

# Global variables for model caching
_processor = None
_model = None
_pipeline = None

def load_model(model_name="Salesforce/blip-image-captioning-base"):
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        _model = BlipForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _processor, _model

def load_pipeline(model_name):
    global _pipeline
    if _pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _pipeline = pipeline(
            "image-text-to-text",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
            token=os.getenv("HF_TOKEN")
        )
    return _pipeline

def analyze_image_with_llm(image_bytes, model="Salesforce/blip-image-captioning-base", custom_prompt=None):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if "google/medgemma" in model:
            pipe = load_pipeline(model)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert medical AI assistant."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": custom_prompt or "Describe this medical image"},
                    {"type": "image", "image": image}
                ]}
            ]
            output = pipe(text=messages, max_new_tokens=500)
            return output[0]["generated_text"][-1]["content"]
        else:
            processor, model_instance = load_model(model)
            inputs = processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = model_instance.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
            return processor.decode(out[0], skip_special_tokens=True)
            
    except Exception as e:
        return f"Fehler bei Bildanalyse: {e}"