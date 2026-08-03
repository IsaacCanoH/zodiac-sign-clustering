from django.urls import path

from .views import (
    KMeansActivateView,
    KMeansDeleteView,
    KMeansExportView,
    KMeansImportView,
    KMeansResetView,
    KMeansResultsPdfView,
    KMeansRetrainView,
    KMeansSaveView,
    KMeansTrainingView,
)


app_name = 'kmeans'

urlpatterns = [
    path('entrenar/', KMeansTrainingView.as_view(), name='train'),
    path('reiniciar/', KMeansResetView.as_view(), name='reset'),
    path('<int:pk>/exportar/', KMeansExportView.as_view(), name='export'),
    path(
        '<int:pk>/resultados-pdf/',
        KMeansResultsPdfView.as_view(),
        name='results_pdf',
    ),
    path('importar/', KMeansImportView.as_view(), name='import'),
    path('<int:pk>/activar/', KMeansActivateView.as_view(), name='activate'),
    path('<int:pk>/reentrenar/', KMeansRetrainView.as_view(), name='retrain'),
    path('<int:pk>/guardar/', KMeansSaveView.as_view(), name='save'),
    path('<int:pk>/eliminar/', KMeansDeleteView.as_view(), name='delete'),
]
