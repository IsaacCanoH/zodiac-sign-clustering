from .forms import DatasetUploadForm
from .services import build_dataset_context


def build_workspace_context(request):
    """Return everything required to render the dataset workspace."""
    context = build_dataset_context(request.GET.get('page'))
    context['dataset_form'] = DatasetUploadForm()
    context['open_upload_modal'] = request.GET.get('upload') == 'invalid'
    return context
