from django.views.generic import TemplateView

from datasets.components import build_workspace_context
from classified_data.components import build_classified_workspace_context
from datasets.model_validation import model_compatibility
from dbscan.components import build_dbscan_workspace_context
from dbscan.models import DBSCANRun
from descriptive_statistics.components import build_statistics_workspace_context
from kmeans.components import build_kmeans_workspace_context
from hierarchical.components import build_hierarchical_workspace_context
from kmeans.models import KMeansRun
from hierarchical.models import HierarchicalRun


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
        context.update(
            build_classified_workspace_context(
                self.request,
                context.get('dataset'),
            )
        )

        # Determine which algorithm should be active in the UI.
        active_algo = context.get('active_algorithm')
        if active_algo == 'dbscan':
            context['ui_active_algorithm'] = 'dbscan'
        elif active_algo == 'hierarchical':
            context['ui_active_algorithm'] = 'hierarchical'
        else:
            context['ui_active_algorithm'] = 'kmeans'

        dataset = context.get('dataset')
        
        # Saved models form an independent catalogue. They must remain visible
        # even when their source dataset is removed or replaced.
        context['all_kmeans_runs'] = list(KMeansRun.objects.filter(is_saved=True))
        context['all_dbscan_runs'] = list(DBSCANRun.objects.filter(is_saved=True))
        context['all_hierarchical_runs'] = list(
            HierarchicalRun.objects.filter(is_saved=True)
        )

        requested_category = context.get('selected_category', '')
        requested_category_column = context.get('category_column', '')
        context['model_requested_category'] = requested_category
        
        context['all_saved_models'] = sorted(
            [
                *[
                    {
                        'algorithm': 'kmeans',
                        'run': run,
                        **model_compatibility(
                            dataset, run,
                            requested_category=requested_category,
                            requested_category_column=requested_category_column,
                        ),
                    }
                    for run in context['all_kmeans_runs']
                ],
                *[
                    {
                        'algorithm': 'dbscan',
                        'run': run,
                        **model_compatibility(
                            dataset, run,
                            requested_category=requested_category,
                            requested_category_column=requested_category_column,
                        ),
                    }
                    for run in context['all_dbscan_runs']
                ],
                *[
                    {
                        'algorithm': 'hierarchical',
                        'run': run,
                        **model_compatibility(
                            dataset, run,
                            requested_category=requested_category,
                            requested_category_column=requested_category_column,
                        ),
                    }
                    for run in context['all_hierarchical_runs']
                ],
            ],
            key=lambda item: item['run'].created_at,
            reverse=True,
        )

        context['saved_model_count'] = len(context['all_saved_models'])

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
            if requested_results_view in {'kmeans', 'dbscan', 'hierarchical'}
            else 'kmeans'
        )
        return context