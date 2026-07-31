from pathlib import Path

from django.http import Http404, HttpResponse
from django.utils.text import slugify
from django.views import View

from datasets.models import Dataset
from datasets.services import filter_dataset_by_category

from .exports import build_statistics_pdf
from .services import build_statistics_context


class StatisticsPdfDownloadView(View):
    def get(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            raise Http404('No hay un dataset disponible.')

        context = build_statistics_context(request, dataset)
        analysis = context['statistics_analysis']
        if not analysis:
            raise Http404('Selecciona una columna válida para generar el reporte.')

        category_filter = filter_dataset_by_category(
            dataset, request.GET.get('category'),
            request.GET.get('category_column'),
        )
        category_label = category_filter['selected_category_label']
        source_name = slugify(Path(dataset.source_name).stem) or 'dataset'
        column_name = slugify(analysis['column']) or 'columna'
        filename = f'estadistica-{source_name}-{column_name}.pdf'
        response = HttpResponse(
            build_statistics_pdf(dataset, analysis, category_label),
            content_type='application/pdf',
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
