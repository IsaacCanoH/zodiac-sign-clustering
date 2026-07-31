from django.urls import path

from .views import (
    DBSCANActivateView,
    DBSCANDeleteView,
    DBSCANExportView,
    DBSCANImportView,
    DBSCANResetView,
    DBSCANRetrainView,
    DBSCANSaveView,
    DBSCANTrainingView,
)


app_name = 'dbscan'

urlpatterns = [
    path('entrenar/', DBSCANTrainingView.as_view(), name='train'),
    path('reiniciar/', DBSCANResetView.as_view(), name='reset'),
    path('<int:pk>/exportar/', DBSCANExportView.as_view(), name='export'),
    path('importar/', DBSCANImportView.as_view(), name='import'),
    path('<int:pk>/activar/', DBSCANActivateView.as_view(), name='activate'),
    path('<int:pk>/reentrenar/', DBSCANRetrainView.as_view(), name='retrain'),
    path('<int:pk>/guardar/', DBSCANSaveView.as_view(), name='save'),
    path('<int:pk>/eliminar/', DBSCANDeleteView.as_view(), name='delete'),
]
