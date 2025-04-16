from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
from PIL import Image
from moviepy import *
import torch

# Load BLIP-2 model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
model.to(device, dtype=dtype)
model.eval()

# Home page view
def index(request):
    return render(request, 'captions/index.html')

# Translation pipeline
def load_translation_pipeline(language):
    try:
        model_name = f"Helsinki-NLP/opus-mt-en-{language}"
        return pipeline("translation", model=model_name)
    except:
        return lambda x: [{"translation_text": x}]

def index(request):
    return render(request, 'captions/index.html')

# Frame extractor for video
def extract_key_frames(video_path, interval=2):
    clip = VideoFileClip(video_path)
    frames = []
    for t in range(0, int(clip.duration), interval):
        frame = clip.get_frame(t)
        frames.append(Image.fromarray(frame).convert("RGB"))
    clip.close()
    return frames

# Image caption
def generate_caption_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, dtype=dtype)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return processor.decode(generated_ids[0], skip_special_tokens=True)

# Frame caption
def generate_caption_from_frame(image):
    inputs = processor(images=image, return_tensors="pt").to(device, dtype=dtype)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return processor.decode(generated_ids[0], skip_special_tokens=True)

# Unified handler
def generate_caption_view(request):
    if request.method == 'POST':
        language = request.POST.get('language', 'en')
        fs = FileSystemStorage()

        if request.FILES.get('image'):
            image = request.FILES['image']
            filename = fs.save(image.name, image)
            path = fs.path(filename)

            try:
                caption = generate_caption_from_image(path)
                if language != 'en':
                    translator = load_translation_pipeline(language)
                    caption = translator(caption)[0]['translation_text']
                return JsonResponse({'status': 'success', 'caption': caption, 'media_url': fs.url(filename)})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)})

        elif request.FILES.get('video'):
            video = request.FILES['video']
            filename = fs.save(video.name, video)
            path = fs.path(filename)

            try:
                frames = extract_key_frames(path)
                captions = [f"{i*2}s: {generate_caption_from_frame(frame)}" for i, frame in enumerate(frames)]
                combined = "\n".join(captions)
                if language != 'en':
                    translator = load_translation_pipeline(language)
                    combined = translator(combined)[0]['translation_text']
                return JsonResponse({'status': 'success', 'caption': combined, 'media_url': fs.url(filename)})
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
