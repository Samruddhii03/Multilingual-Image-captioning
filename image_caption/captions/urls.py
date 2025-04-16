from django.urls import path
# from .views.views1 import index
from .views.views2 import generate_caption_view,index

urlpatterns = [
    path('', index, name='index'),
    path('generate-caption/', generate_caption_view, name='generate_caption_view'),
    # path('generate-caption/', generate_caption_view, name='generate_caption_view'),
    # path('generate_video_caption/', generate_video_caption, name='generate_video_caption'),
]
