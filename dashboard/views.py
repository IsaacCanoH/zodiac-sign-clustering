from django.views.generic import TemplateView

from datasets.components import build_workspace_context
from dbscan.components import build_dbscan_workspace_context
from descriptive_statistics.components import build_statistics_workspace_context
from kmeans.components import build_kmeans_workspace_context
from hierarchical.components import build_hierarchical_workspace_context # <--- NUEVO IMPORT


class DashboardView(TemplateView):
    template_name = 'dashboard/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(build_workspace_context(self.request))
        context.update(
            build_statistics_workspace_context(
                self.request, context.get('dataset')
            )
        )
        context.update(
            build_kmeans_workspace_context(
                self.request, context.get('dataset')
            )
        )
        context.update(
            build_dbscan_workspace_context(
                self.request, context.get('dataset')
            )
        )
        context.update(
            build_hierarchical_workspace_context(
                self.request, context.get('dataset')
            )
        )

        active_algo = context.get('active_algorithm')
        if active_algo == 'dbscan':
            context['ui_active_algorithm'] = 'dbscan'
        elif active_algo == 'hierarchical':
            context['ui_active_algorithm'] = 'hierarchical'
        else:
            context['ui_active_algorithm'] = 'kmeans'

        dataset = context.get('dataset')
        context['all_kmeans_runs'] = (
            list(dataset.kmeans_runs.all()) if dataset else []
        )
        context['all_dbscan_runs'] = (
            list(dataset.dbscan_runs.all()) if dataset else []
        )
        context['all_hierarchical_runs'] = (
            list(dataset.hierarchical_runs.all()) if dataset else []
        )

        context['model_import_error'] = self.request.session.pop(
            'model_import_error', None
        )
        return context