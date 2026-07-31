from django.views.generic import TemplateView

from datasets.components import build_workspace_context
from classified_data.components import build_classified_workspace_context
from datasets.model_validation import dataset_fingerprint
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
        context.update(
            build_classified_workspace_context(
                self.request,
                context.get('dataset'),
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
        current_fingerprint = dataset_fingerprint(dataset) if dataset else ''
        context['all_saved_models'] = sorted(
            [
                *[
                    {
                        'algorithm': 'kmeans',
                        'run': run,
                        'compatible': (
                            run.dataset_fingerprint == current_fingerprint
                        ),
                    }
                    for run in context['all_kmeans_runs']
                ],
                *[
                    {
                        'algorithm': 'dbscan',
                        'run': run,
                        'compatible': (
                            run.dataset_fingerprint == current_fingerprint
                        ),
                    }
                    for run in context['all_dbscan_runs']
                ],
            ],
            key=lambda item: item['run'].created_at,
            reverse=True,
        )

        # Import error message (set by import views via session)
        context['model_import_error'] = self.request.session.pop(
            'model_import_error', None
        )
        context['model_action_error'] = self.request.session.pop(
            'model_action_error',
            None,
        )
        requested_results_view = self.request.GET.get('results_view')
        context['results_view'] = (
            requested_results_view
            if requested_results_view in {'kmeans', 'dbscan'}
            else 'kmeans'
        )
        return context

