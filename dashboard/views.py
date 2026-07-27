from django.views.generic import TemplateView

from datasets.components import build_workspace_context
from descriptive_statistics.components import build_statistics_workspace_context


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
        return context
