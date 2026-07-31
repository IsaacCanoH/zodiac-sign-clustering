from django.views.generic import TemplateView

from datasets.components import build_workspace_context
from dbscan.components import build_dbscan_workspace_context
from descriptive_statistics.components import build_statistics_workspace_context
from kmeans.components import build_kmeans_workspace_context


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
        # Determine which algorithm should be active in the UI.
        if context.get('active_algorithm') == 'dbscan':
            context['ui_active_algorithm'] = 'dbscan'
        else:
            context['ui_active_algorithm'] = 'kmeans'

        # All saved runs for the models pane
        dataset = context.get('dataset')
        context['all_kmeans_runs'] = (
            list(dataset.kmeans_runs.all()) if dataset else []
        )
        context['all_dbscan_runs'] = (
            list(dataset.dbscan_runs.all()) if dataset else []
        )

        # Import error message (set by import views via session)
        context['model_import_error'] = self.request.session.pop(
            'model_import_error', None
        )
        return context

