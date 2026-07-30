from django.urls import path

from .views import KMeansResetView, KMeansTrainingView


app_name = 'kmeans'

urlpatterns = [
    path('entrenar/', KMeansTrainingView.as_view(), name='train'),
    path('reiniciar/', KMeansResetView.as_view(), name='reset'),
]
