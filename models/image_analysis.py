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
        elif torch.backends.mps.is_available():
            _model = _model.to("mps")
    return _processor, _model

def load_pipeline(model_name):
    global _pipeline
    if _pipeline is None:
        # MedGemma funktioniert besser auf CPU
        if "google/medgemma" in model_name:
            device = "cpu"
            torch_dtype = torch.float32
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
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
        debug = os.getenv("DEBUG", "false").lower() == "true"
        
        if debug:
            print(f"Debug: Analyzing with model {model}")
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if debug:
            print(f"Debug: Image loaded, size: {image.size}")
        
        if "google/medgemma" in model:
            if debug:
                print("Debug: Using MedGemma pipeline")
            pipe = load_pipeline(model)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert medical AI assistant."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": custom_prompt or "Describe this medical image"},
                    {"type": "image", "image": image}
                ]}
            ]
            if debug:
                print("Debug: Generating response...")
            output = pipe(
                text=messages, 
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            if debug:
                print(f"Debug: Output received: {type(output)}")
                print(f"Debug: Full output: {output}")
            
            # Extrahiere die Assistant-Antwort
            if (isinstance(output, list) and len(output) > 0 and 
                "generated_text" in output[0] and 
                isinstance(output[0]["generated_text"], list)):
                
                generated_text = output[0]["generated_text"]
                # Finde die Assistant-Nachricht
                for msg in generated_text:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content.strip():
                            return content
                        else:
                            return "Das Modell hat keine Antwort generiert. Versuchen Sie einen anderen Prompt."
            
            return "Fehler beim Extrahieren der Antwort."
        else:
            if debug:
                print("Debug: Using BLIP model")
            processor, model_instance = load_model(model)
            inputs = processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            with torch.no_grad():
                out = model_instance.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
            result = processor.decode(out[0], skip_special_tokens=True)
            if debug:
                print(f"Debug: BLIP result: {result}")
            return result
            
    except Exception as e:
        error_msg = f"Fehler bei Bildanalyse: {e}"
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"Debug: Exception occurred: {error_msg}")
        return error_msg