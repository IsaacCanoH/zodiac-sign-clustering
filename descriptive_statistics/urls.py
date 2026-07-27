from django.urls import path

from .views import StatisticsPdfDownloadView


app_name = 'descriptive_statistics'

urlpatterns = [
    path('descargar-pdf/', StatisticsPdfDownloadView.as_view(), name='download_pdf'),
]
