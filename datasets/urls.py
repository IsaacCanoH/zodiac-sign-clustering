from django.urls import path

from .views import (
    DatasetDeleteView,
    DatasetUploadView,
    EquivalenceDeleteView,
    EquivalenceRemoveApplicationView,
    EquivalenceSaveView,
    FilteredDatasetDownloadView,
)

app_name = 'datasets'

urlpatterns = [
    path('cargar/', DatasetUploadView.as_view(), name='upload'),
    path('eliminar/', DatasetDeleteView.as_view(), name='delete'),
    path('descargar/', FilteredDatasetDownloadView.as_view(), name='download'),
    path('equivalencias/guardar/', EquivalenceSaveView.as_view(), name='equivalence_save'),
    path(
        'equivalencias/<int:configuration_id>/eliminar/',
        EquivalenceDeleteView.as_view(),
        name='equivalence_delete',
    ),
    path(
        'equivalencias/<int:configuration_id>/quitar/',
        EquivalenceRemoveApplicationView.as_view(),
        name='equivalence_remove_application',
    ),
]
