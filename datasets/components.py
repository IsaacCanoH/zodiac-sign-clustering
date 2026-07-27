from .forms import DatasetUploadForm
from .services import build_dataset_context, build_equivalence_context


def build_workspace_context(request):
    """Return everything required to render the dataset workspace."""
    context = build_dataset_context(
        page_number=request.GET.get('page'),
        requested_category=request.GET.get('category'),
        requested_representation=request.GET.get('representation', 'original'),
    )
    context['equivalence_data'] = build_equivalence_context(context.get('dataset'))
    context['dataset_form'] = DatasetUploadForm()
    context['open_upload_modal'] = request.GET.get('upload') == 'invalid'
    return context
