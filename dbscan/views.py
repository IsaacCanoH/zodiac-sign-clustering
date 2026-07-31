import json

from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import View

from datasets.models import Dataset

from .exports import export_dbscan_run, import_dbscan_run
from .forms import DBSCANTrainingForm
from .models import DBSCANRun
from .services import (
    DBSCANTrainingError,
    build_dbscan_training_setup,
    clear_dbscan_runs,
    train_dbscan,
)


class DBSCANTrainingView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            return redirect(f"{reverse('dashboard:index')}#training-pane")

        setup = build_dbscan_training_setup(
            dataset, request.POST.get('category')
        )
        form = DBSCANTrainingForm(
            request.POST,
            numeric_columns=setup['dbscan_numeric_columns'],
            categorical_columns=setup['dbscan_categorical_columns'],
        )
        if not form.is_valid():
            request.session['dbscan_form_state'] = {
                'data': {
                    'algorithm': 'dbscan',
                    'epsilon': request.POST.get('epsilon', ''),
                    'min_samples': request.POST.get('min_samples', ''),
                    'columns': request.POST.getlist('columns'),
                    'comparison_column': request.POST.get(
                        'comparison_column', ''
                    ),
                },
                'errors': form.errors.get_json_data(),
            }
            return redirect(f"{reverse('dashboard:index')}#training-pane")

        try:
            train_dbscan(
                dataset=dataset,
                selected_columns=form.cleaned_data['columns'],
                epsilon=form.cleaned_data['epsilon'],
                min_samples=form.cleaned_data['min_samples'],
                requested_category=request.POST.get('category'),
                comparison_column=form.cleaned_data['comparison_column'],
            )
        except DBSCANTrainingError as error:
            request.session['dbscan_form_state'] = {
                'data': {
                    'algorithm': 'dbscan',
                    'epsilon': form.cleaned_data['epsilon'],
                    'min_samples': form.cleaned_data['min_samples'],
                    'columns': form.cleaned_data['columns'],
                    'comparison_column': form.cleaned_data['comparison_column'],
                },
                'errors': {'__all__': [{'message': str(error), 'code': ''}]},
            }
            return redirect(f"{reverse('dashboard:index')}#training-pane")

        return redirect(f"{reverse('dashboard:index')}#results-pane")


class DBSCANResetView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            clear_dbscan_runs(dataset)
        request.session.pop('dbscan_form_state', None)
        return redirect(f"{reverse('dashboard:index')}#training-pane")


class DBSCANExportView(View):
    """Download a single DBSCANRun as a JSON file."""

    def get(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        run = get_object_or_404(DBSCANRun, pk=pk, dataset=dataset)
        data = export_dbscan_run(run)
        filename = (
            f"dbscan_eps{run.epsilon}_min{run.min_samples}"
            f"_{run.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        )
        response = HttpResponse(
            json.dumps(data, ensure_ascii=False, indent=2),
            content_type='application/json',
        )
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


class DBSCANImportView(View):
    """Upload a JSON file and restore it as a new DBSCANRun."""

    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            request.session['model_import_error'] = 'No hay un dataset cargado.'
            return redirect(f"{reverse('dashboard:index')}#models-pane")

        uploaded = request.FILES.get('model_file')
        if not uploaded:
            request.session['model_import_error'] = 'No se seleccionó ningún archivo.'
            return redirect(f"{reverse('dashboard:index')}#models-pane")

        try:
            raw = uploaded.read().decode('utf-8')
            data = json.loads(raw)
            import_dbscan_run(dataset, data)
        except (json.JSONDecodeError, KeyError, TypeError):
            request.session['model_import_error'] = (
                'El archivo no es un modelo DBSCAN válido o está corrupto.'
            )
            return redirect(f"{reverse('dashboard:index')}#models-pane")
        except ValueError as error:
            request.session['model_import_error'] = str(error)
            return redirect(f"{reverse('dashboard:index')}#models-pane")

        return redirect(f"{reverse('dashboard:index')}#results-pane")


class DBSCANActivateView(View):
    """Make a saved DBSCANRun the active (most-recent) result."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            DBSCANRun.objects.filter(pk=pk, dataset=dataset).update(
                created_at=timezone.now()
            )
        return redirect(f"{reverse('dashboard:index')}#results-pane")


class DBSCANDeleteView(View):
    """Delete a single saved DBSCANRun."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            DBSCANRun.objects.filter(pk=pk, dataset=dataset).delete()
        return redirect(f"{reverse('dashboard:index')}#models-pane")

