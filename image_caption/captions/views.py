from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from django.shortcuts import render

# Load BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Translation pipeline using Hugging Face
def load_translation_pipeline(language):
    model_name = f"Helsinki-NLP/opus-mt-en-{language}"
    return pipeline("translation", model=model_name)

def index(request):
    return render(request, 'captions/index.html')

def generate_caption_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        language = request.POST.get('language', 'en')

        # Save the image
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        image_path = fs.path(filename)

        try:
            # Generate Caption in English
            caption = generate_caption(image_path)

            # Translate if not English
            if language != 'en':
                translator = load_translation_pipeline(language)
                translation = translator(caption)[0]['translation_text']
            else:
                translation = caption

            return JsonResponse({'status': 'success', 'caption': translation, 'image_url': fs.url(filename)})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
