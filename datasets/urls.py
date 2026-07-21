from django.urls import path

from .views import DatasetDeleteView, DatasetUploadView

app_name = 'datasets'

urlpatterns = [
    path('cargar/', DatasetUploadView.as_view(), name='upload'),
    path('eliminar/', DatasetDeleteView.as_view(), name='delete'),
]
