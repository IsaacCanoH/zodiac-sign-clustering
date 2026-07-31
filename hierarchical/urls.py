from django.urls import path

from .views import (
    HierarchicalActivateView,
    HierarchicalDeleteView,
    HierarchicalExportView,
    HierarchicalImportView,
    HierarchicalResetView,
    HierarchicalRetrainView,
    HierarchicalSaveView,
    HierarchicalTrainingView,
)


app_name = 'hierarchical'

urlpatterns = [
    path('entrenar/', HierarchicalTrainingView.as_view(), name='train'),
    path('reiniciar/', HierarchicalResetView.as_view(), name='reset'),
    path('<int:pk>/exportar/', HierarchicalExportView.as_view(), name='export'),
    path('importar/', HierarchicalImportView.as_view(), name='import'),
    path('<int:pk>/activar/', HierarchicalActivateView.as_view(), name='activate'),
    path('<int:pk>/reentrenar/', HierarchicalRetrainView.as_view(), name='retrain'),
    path('<int:pk>/guardar/', HierarchicalSaveView.as_view(), name='save'),
    path('<int:pk>/eliminar/', HierarchicalDeleteView.as_view(), name='delete'),
]