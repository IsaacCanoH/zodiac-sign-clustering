from django.views.generic import TemplateView

from datasets.components import build_workspace_context


class DashboardView(TemplateView):
    template_name = 'dashboard/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(build_workspace_context(self.request))
        return context
