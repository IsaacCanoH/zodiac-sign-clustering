import json

from django.core.exceptions import ValidationError
from django.db import DatabaseError
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import View

from datasets.models import Dataset
from datasets.model_validation import dataset_fingerprint

from .exports import export_kmeans_run, import_kmeans_run
from .forms import KMeansTrainingForm
from .models import KMeansRun
from .services import (
    KMeansTrainingError,
    build_training_setup,
    clear_kmeans_runs,
    train_kmeans,
)


def _dashboard_redirect(fragment, **query):
    clean_query = {
        key: value for key, value in query.items() if value not in (None, '')
    }
    return redirect(
        reverse('dashboard:index', query=clean_query, fragment=fragment)
    )


class KMeansTrainingView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            return _dashboard_redirect(
                'training-pane',
                category=request.POST.get('category'),
            )

        setup = build_training_setup(dataset, request.POST.get('category'))
        form = KMeansTrainingForm(
            request.POST,
            numeric_columns=setup['numeric_columns'],
            categorical_columns=setup['categorical_columns'],
            max_clusters=max(setup['max_clusters'], 2),
        )
        if not form.is_valid():
            request.session['kmeans_form_state'] = {
                'data': {
                    'algorithm': request.POST.get('algorithm', 'kmeans'),
                    'cluster_count': request.POST.get('cluster_count', ''),
                    'columns': request.POST.getlist('columns'),
                    'comparison_column': request.POST.get(
                        'comparison_column', ''
                    ),
                },
                'errors': form.errors.get_json_data(),
            }
            return _dashboard_redirect(
                'training-pane',
                category=request.POST.get('category'),
            )

        try:
            train_kmeans(
                dataset=dataset,
                selected_columns=form.cleaned_data['columns'],
                cluster_count=form.cleaned_data['cluster_count'],
                requested_category=request.POST.get('category'),
                comparison_column=form.cleaned_data['comparison_column'],
            )
        except KMeansTrainingError as error:
            request.session['kmeans_form_state'] = {
                'data': {
                    'algorithm': 'kmeans',
                    'cluster_count': form.cleaned_data['cluster_count'],
                    'columns': form.cleaned_data['columns'],
                    'comparison_column': form.cleaned_data[
                        'comparison_column'
                    ],
                },
                'errors': {'__all__': [{'message': str(error), 'code': ''}]},
            }
            return _dashboard_redirect(
                'training-pane',
                category=request.POST.get('category'),
            )

        return _dashboard_redirect('results-pane', results_view='kmeans')


class KMeansResetView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            clear_kmeans_runs(dataset)
        request.session.pop('kmeans_form_state', None)
        return _dashboard_redirect('training-pane')


class KMeansExportView(View):
    """Download a single KMeansRun as a JSON file."""

    def get(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        run = get_object_or_404(KMeansRun, pk=pk, dataset=dataset)
        data = export_kmeans_run(run)
        filename = (
            f"kmeans_{run.cluster_count}clusters"
            f"_{run.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        response = HttpResponse(
            json.dumps(data, ensure_ascii=False, indent=2),
            content_type='application/json',
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


class KMeansImportView(View):
    """Upload a JSON file and restore it as a new KMeansRun."""

    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            request.session['model_import_error'] = 'No hay un dataset cargado.'
            return _dashboard_redirect('models-pane')

        uploaded = request.FILES.get('model_file')
        if not uploaded:
            request.session['model_import_error'] = 'No se seleccionó ningún archivo.'
            return _dashboard_redirect('models-pane')
        if uploaded.size > 5 * 1024 * 1024:
            request.session['model_import_error'] = (
                'El archivo supera el límite permitido de 5 MB.'
            )
            return _dashboard_redirect('models-pane')

        try:
            raw = uploaded.read().decode('utf-8')
            data = json.loads(raw)
            import_kmeans_run(dataset, data)
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            KeyError,
            TypeError,
            ValidationError,
            DatabaseError,
        ):
            request.session['model_import_error'] = (
                'El archivo no es un modelo K-Means válido o está corrupto.'
            )
            return _dashboard_redirect('models-pane')
        except ValueError as error:
            request.session['model_import_error'] = str(error)
            return _dashboard_redirect('models-pane')

        return _dashboard_redirect('results-pane', results_view='kmeans')


class KMeansActivateView(View):
    """Make a saved KMeansRun the active (most-recent) result."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            run = get_object_or_404(KMeansRun, pk=pk, dataset=dataset)
            if run.dataset_fingerprint != dataset_fingerprint(dataset):
                request.session['model_action_error'] = (
                    'Este modelo K-Means pertenece a otro dataset. '
                    'Carga el dataset original para activarlo.'
                )
                return _dashboard_redirect('models-pane')
            run.activated_at = timezone.now()
            run.save(update_fields=('activated_at',))
        return _dashboard_redirect('results-pane', results_view='kmeans')


class KMeansDeleteView(View):
    """Delete a single saved KMeansRun."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            KMeansRun.objects.filter(pk=pk, dataset=dataset).delete()
        return _dashboard_redirect('models-pane')

