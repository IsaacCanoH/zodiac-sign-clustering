from django.urls import path

from .views import ClassifiedDataDownloadView


app_name = 'classified_data'

urlpatterns = [
    path('descargar/', ClassifiedDataDownloadView.as_view(), name='download'),
]
