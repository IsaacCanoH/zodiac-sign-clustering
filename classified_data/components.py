from .services import build_classified_context


def build_classified_workspace_context(request, dataset):
    return build_classified_context(
        dataset,
        requested_algorithm=request.GET.get('classified_algorithm'),
        requested_cluster=request.GET.get('classified_cluster'),
        page_number=request.GET.get('classified_page'),
    )
