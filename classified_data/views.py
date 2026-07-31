from pathlib import Path

from django.http import Http404, HttpResponse
from django.utils.text import slugify
from django.views import View

from datasets.exports import build_excel_file_from_rows
from datasets.models import Dataset

from .services import ALGORITHMS, build_classified_records


class ClassifiedDataDownloadView(View):
    def get(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            raise Http404('No hay un dataset disponible.')

        algorithm = request.GET.get('algorithm', '')
        if algorithm not in ALGORITHMS:
            raise Http404('Selecciona un algoritmo válido.')
        classified = build_classified_records(
            dataset,
            algorithm,
            request.GET.get('cluster'),
        )
        if not classified['run']:
            raise Http404('No existe un entrenamiento para este algoritmo.')
        if classified['invalid_cluster']:
            raise Http404('El filtro de cluster no es válido.')

        config = ALGORITHMS[algorithm]
        headers = [
            'Número de fila original',
            config['cluster_column'],
            *dataset.columns,
        ]
        rows = [
            [row['row_number'], row['cluster_label'], *row['values']]
            for row in classified['records']
        ]
        source_name = slugify(Path(dataset.source_name).stem) or 'dataset'
        filter_suffix = (
            f"-{slugify(classified['selected_cluster'])}"
            if classified['selected_cluster']
            else ''
        )
        filename = (
            f'{source_name}-datos-clasificados-{algorithm}{filter_suffix}.xlsx'
        )
        response = HttpResponse(
            build_excel_file_from_rows(
                headers,
                rows,
                worksheet_title='Datos clasificados',
                table_name='DatosClasificados',
            ),
            content_type=(
                'application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.sheet'
            ),
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
