from .services import build_statistics_context


def build_statistics_workspace_context(request, dataset):
    """Provide the descriptive-statistics component with isolated context."""
    return build_statistics_context(request, dataset)
