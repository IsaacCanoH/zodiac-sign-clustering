import json
from pathlib import Path

from django.contrib import messages
from django.db import transaction
from django.http import Http404, HttpResponse
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.text import slugify
from django.views import View

from .forms import DatasetUploadForm
from .exports import build_excel_file
from .equivalences import transform_records, validate_equivalence_payload
from .models import (
    Dataset,
    DatasetEquivalenceApplication,
    EquivalenceConfiguration,
)
from .services import (
    filter_dataset_by_category,
    remove_dataset,
    replace_dataset,
    generate_synthetic_records,
)


class DatasetUploadView(View):
    def post(self, request):
        form = DatasetUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            for error in form.errors.get('__all__', []) + form.errors.get('file', []):
                messages.error(request, error)
            return redirect(f"{reverse('dashboard:index')}?upload=invalid")

        replace_dataset(form.cleaned_data)
        messages.success(request, 'Conjunto de datos cargado correctamente. Ya puedes configurar el análisis.')
        return redirect('dashboard:index')


class DatasetDeleteView(View):
    def post(self, request):
        remove_dataset()
        messages.success(request, 'Conjunto de datos eliminado correctamente.')
        return redirect('dashboard:index')


class EquivalenceSaveView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            return JsonResponse({'errors': {'dataset': 'No hay un dataset cargado.'}}, status=400)

        try:
            payload = json.loads(request.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({'errors': {'form': 'La solicitud no es válida.'}}, status=400)
        if not isinstance(payload, dict):
            return JsonResponse(
                {'errors': {'form': 'La solicitud no es válida.'}},
                status=400,
            )

        cleaned_data, errors = validate_equivalence_payload(dataset, payload)
        if errors:
            return JsonResponse({'errors': errors}, status=400)

        configuration_id = payload.get('configuration_id')
        if configuration_id not in (None, ''):
            try:
                configuration_id = int(configuration_id)
            except (TypeError, ValueError):
                return JsonResponse(
                    {'errors': {'configuration': 'La configuración no es válida.'}},
                    status=400,
                )
        with transaction.atomic():
            if configuration_id:
                configuration = EquivalenceConfiguration.objects.filter(
                    pk=configuration_id
                ).first()
                if not configuration:
                    return JsonResponse(
                        {'errors': {'configuration': 'La configuración no existe.'}},
                        status=404,
                    )
            else:
                configuration = EquivalenceConfiguration()

            configuration.name = cleaned_data['name']
            configuration.mapping = cleaned_data['mapping']
            configuration.possible_values = cleaned_data['possible_values']
            configuration.source_dataset_name = dataset.source_name
            configuration.save()

            selected_columns = set(cleaned_data['columns'])
            other_applications = dataset.equivalence_applications.exclude(
                configuration=configuration
            )
            for application in other_applications:
                remaining_columns = [
                    column
                    for column in application.columns
                    if column not in selected_columns
                ]
                if remaining_columns:
                    application.columns = remaining_columns
                    application.save(update_fields=('columns', 'updated_at'))
                else:
                    application.delete()

            DatasetEquivalenceApplication.objects.update_or_create(
                dataset=dataset,
                configuration=configuration,
                defaults={'columns': cleaned_data['columns']},
            )

        return JsonResponse(
            {
                'message': 'La configuración se guardó y aplicó correctamente.',
                'configuration_id': configuration.pk,
            }
        )


class EquivalenceDeleteView(View):
    def post(self, request, configuration_id):
        configuration = EquivalenceConfiguration.objects.filter(
            pk=configuration_id
        ).first()
        if not configuration:
            raise Http404('La configuración no existe.')
        configuration.delete()
        return JsonResponse({'message': 'La configuración se eliminó correctamente.'})


class EquivalenceRemoveApplicationView(View):
    def post(self, request, configuration_id):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            raise Http404('No hay un dataset disponible.')
        DatasetEquivalenceApplication.objects.filter(
            dataset=dataset, configuration_id=configuration_id
        ).delete()
        return JsonResponse({'message': 'La configuración se quitó del dataset.'})


class FilteredDatasetDownloadView(View):
    def get(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            raise Http404('No hay un dataset disponible.')

        category_filter = filter_dataset_by_category(
            dataset, request.GET.getlist('category'),
            request.GET.get('category_column'),
        )
        selected_category = category_filter['selected_category']
        selected_categories = category_filter['selected_categories']

        representation = request.GET.get('representation', 'original')
        columns, records, active_representation = transform_records(
            dataset, category_filter['filtered_records'], representation
        )
        source_name = slugify(Path(dataset.source_name).stem) or 'dataset'
        if selected_category:
            filename = f'{source_name}-{slugify(selected_category)}-{active_representation}.xlsx'
        elif selected_categories:
            filename = f'{source_name}-filtrado-{active_representation}.xlsx'
        else:
            filename = f'{source_name}-completo-{active_representation}.xlsx'
            
        response = HttpResponse(
            build_excel_file(columns, records),
            content_type=(
                'application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.sheet'
            ),
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'

        return response


class GenerateSyntheticDataView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            messages.error(request, 'No hay un dataset disponible para generar datos.')
            return redirect('dashboard:index')

        try:
            num_records = int(request.POST.get('num_records', 0))
        except ValueError:
            num_records = 0
            
        pivot_column = request.POST.get('pivot_column', '')
        
        try:
            noise_level = float(request.POST.get('noise_level', 0.5))
        except ValueError:
            noise_level = 0.5

        if num_records <= 0:
            messages.error(request, 'Cantidad de registros inválida.')
            return redirect('dashboard:index')

        success = generate_synthetic_records(dataset, num_records, pivot_column, noise_level)
        if success:
            messages.success(request, f'Se generaron exitosamente {num_records} registros sintéticos.')
        else:
            messages.error(request, 'Ocurrió un error al generar los datos sintéticos.')
            
        return redirect('dashboard:index')
