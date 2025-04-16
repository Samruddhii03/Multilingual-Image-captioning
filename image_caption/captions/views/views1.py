# from django.http import JsonResponse
# from django.core.files.storage import FileSystemStorage
# from django.shortcuts import render
# from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
# from PIL import Image
# import torch

# # Load BLIP-2 Processor and Model
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
# model.eval()

# # Load translation pipeline dynamically
# def load_translation_pipeline(language):
#     try:
#         model_name = f"Helsinki-NLP/opus-mt-en-{language}"
#         return pipeline("translation", model=model_name)
#     except Exception as e:
#         print(f"[Translation Error] Failed to load model for '{language}': {e}")
#         return lambda x: [{"translation_text": x}]

# # Home page view
# def index(request):
#     return render(request, 'captions/index.html')

# # Caption generation view
# def generate_caption_view(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         image = request.FILES['image']
#         language = request.POST.get('language', 'en')

#         fs = FileSystemStorage()
#         filename = fs.save(image.name, image)
#         image_path = fs.path(filename)

#         try:
#             caption = generate_caption(image_path)

#             if language != 'en':
#                 translator = load_translation_pipeline(language)
#                 result = translator(caption)
#                 translation = result[0]['translation_text'] if isinstance(result, list) else caption
#             else:
#                 translation = caption

#             return JsonResponse({
#                 'status': 'success',
#                 'caption': translation,
#                 'image_url': fs.url(filename)
#             })

#         except Exception as e:
#             return JsonResponse({'status': 'error', 'message': str(e)})

#     return JsonResponse({'status': 'error', 'message': 'Invalid request'})

# # Generate caption from image using BLIP-2
# def generate_caption(image_path):
#     image = Image.open(image_path).convert("RGB")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float16 if device == "cuda" else torch.float32
#     model.to(device, dtype=dtype)

#     inputs = processor(images=image, return_tensors="pt").to(device, dtype=dtype)

#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         num_beams=5,
#         no_repeat_ngram_size=2,
#         early_stopping=True
#     )

#     caption = processor.decode(generated_ids[0], skip_special_tokens=True)
#     return caption

