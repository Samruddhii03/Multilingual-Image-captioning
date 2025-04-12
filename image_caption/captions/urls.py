from django.urls import path
from .views import index, generate_caption_view

urlpatterns = [
    path('', index, name='index'),
    path('generate-caption/', generate_caption_view, name='generate_caption_view'),
]
