from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse
from django.views import View

from .forms import DatasetUploadForm
from .services import remove_dataset, replace_dataset


class DatasetUploadView(View):
    def post(self, request):
        form = DatasetUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            for error in form.errors.get('__all__', []) + form.errors.get('file', []):
                messages.error(request, error)
            return redirect(f"{reverse('dashboard:index')}?upload=invalid")

        replace_dataset(form.cleaned_data)
        messages.success(request, 'El dataset se cargó correctamente.')
        return redirect('dashboard:index')


class DatasetDeleteView(View):
    def post(self, request):
        remove_dataset()
        messages.success(request, 'El dataset se eliminó correctamente.')
        return redirect('dashboard:index')
