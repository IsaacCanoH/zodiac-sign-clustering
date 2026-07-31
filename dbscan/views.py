import json

from django.core.exceptions import ValidationError
from django.contrib import messages
from django.db import DatabaseError
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import View

from datasets.models import Dataset
from datasets.model_validation import model_compatibility

from .exports import export_dbscan_run, import_dbscan_run
from .forms import DBSCANSaveForm, DBSCANTrainingForm
from .models import DBSCANRun
from .services import (
    DBSCANTrainingError,
    build_dbscan_training_setup,
    clear_dbscan_runs,
    train_dbscan,
)


def _dashboard_redirect(fragment, **query):
    clean_query = {
        key: value for key, value in query.items() if value not in (None, '')
    }
    return redirect(
        reverse('dashboard:index', query=clean_query, fragment=fragment)
    )


class DBSCANTrainingView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            return _dashboard_redirect(
                'training-pane',
                category=request.POST.get('category'),
            )

        setup = build_dbscan_training_setup(
            dataset, request.POST.get('category'),
            request.POST.get('category_column'),
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
                    'name': request.POST.get('name', ''),
                    'topic': request.POST.get('topic', ''),
                    'description': request.POST.get('description', ''),
                    'epsilon': request.POST.get('epsilon', ''),
                    'min_samples': request.POST.get('min_samples', ''),
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
            train_dbscan(
                dataset=dataset,
                selected_columns=form.cleaned_data['columns'],
                epsilon=form.cleaned_data['epsilon'],
                min_samples=form.cleaned_data['min_samples'],
                requested_category=request.POST.get('category'),
                requested_category_column=request.POST.get('category_column'),
                comparison_column=form.cleaned_data['comparison_column'],
                name=form.cleaned_data['name'],
                topic=form.cleaned_data['topic'],
                description=form.cleaned_data['description'],
                save_immediately=False,
            )
        except DBSCANTrainingError as error:
            request.session['dbscan_form_state'] = {
                'data': {
                    'algorithm': 'dbscan',
                    'name': form.cleaned_data['name'],
                    'topic': form.cleaned_data['topic'],
                    'description': form.cleaned_data['description'],
                    'epsilon': form.cleaned_data['epsilon'],
                    'min_samples': form.cleaned_data['min_samples'],
                    'columns': form.cleaned_data['columns'],
                    'comparison_column': form.cleaned_data['comparison_column'],
                },
                'errors': {'__all__': [{'message': str(error), 'code': ''}]},
            }
            return _dashboard_redirect(
                'training-pane',
                category=request.POST.get('category'),
            )

        return _dashboard_redirect('results-pane', results_view='dbscan')


class DBSCANResetView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            clear_dbscan_runs(dataset)
        request.session.pop('dbscan_form_state', None)
        return _dashboard_redirect('training-pane')


class DBSCANExportView(View):
    """Download a single DBSCANRun as a JSON file."""

    def get(self, request, pk):
        run = get_object_or_404(DBSCANRun, pk=pk)
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
            import_dbscan_run(
                dataset, data,
                allow_compatible=request.POST.get('import_mode') == 'retrain',
            )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            KeyError,
            TypeError,
            ValidationError,
            DatabaseError,
        ):
            request.session['model_import_error'] = (
                'El archivo no es un modelo DBSCAN válido o está corrupto.'
            )
            return _dashboard_redirect('models-pane')
        except ValueError as error:
            request.session['model_import_error'] = str(error)
            return _dashboard_redirect('models-pane')

        if request.POST.get('import_mode') == 'retrain':
            messages.success(
                request,
                'Modelo compatible importado. Ya puedes reentrenarlo desde el catálogo.',
            )
            return _dashboard_redirect('models-pane')
        return _dashboard_redirect('results-pane', results_view='dbscan')


class DBSCANActivateView(View):
    """Make a saved DBSCANRun the active (most-recent) result."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        run = get_object_or_404(DBSCANRun, pk=pk)
        if not run.is_saved:
            request.session['model_action_error'] = (
                'Guarda el modelo antes de crear una versión reentrenada.'
            )
            return _dashboard_redirect('models-pane')
        requested_category = request.POST.get('category', run.category_filter)
        requested_category_column = request.POST.get(
            'category_column', run.category_column
        )
        compatibility = model_compatibility(
            dataset, run, requested_category=requested_category,
            requested_category_column=requested_category_column,
        )
        if not compatibility['exact'] or not compatibility['compatible']:
            reasons = compatibility['reasons'] or [
                'Activar requiere exactamente el dataset con el que se entrenó.'
            ]
            if not compatibility['exact']:
                reasons.insert(0, 'Este modelo pertenece a otro dataset.')
            request.session['model_action_error'] = (
                'No se puede activar: ' + ' '.join(reasons)
            )
            return _dashboard_redirect('models-pane')
        run.dataset = dataset
        run.activated_at = timezone.now()
        run.save(update_fields=('dataset', 'activated_at'))
        return _dashboard_redirect('results-pane', results_view='dbscan')


class DBSCANRetrainView(View):
    """Re-run DBSCAN on current compatible data and retain its lineage."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        run = get_object_or_404(DBSCANRun, pk=pk)
        requested_category = request.POST.get('category', run.category_filter)
        requested_category_column = request.POST.get(
            'category_column', run.category_column
        )
        compatibility = model_compatibility(
            dataset, run, requested_category=requested_category,
            requested_category_column=requested_category_column,
        )
        if not compatibility['compatible']:
            request.session['model_action_error'] = (
                'No se puede reentrenar: ' + ' '.join(compatibility['reasons'])
            )
            return _dashboard_redirect('models-pane')
        try:
            train_dbscan(
                dataset, run.selected_columns, run.epsilon, run.min_samples,
                requested_category=requested_category,
                requested_category_column=run.category_column,
                comparison_column=run.comparison_column,
                name=run.name, topic=run.topic, description=run.description,
                parent_run=run,
                save_immediately=False,
            )
        except DBSCANTrainingError as error:
            request.session['model_action_error'] = str(error)
            return _dashboard_redirect('models-pane')
        return _dashboard_redirect('results-pane', results_view='dbscan')


class DBSCANSaveView(View):
    """Promote a provisional DBSCAN result to the saved catalogue."""

    def post(self, request, pk):
        dataset = Dataset.objects.filter(pk=1).first()
        run = get_object_or_404(DBSCANRun, pk=pk, dataset=dataset)
        if run.is_saved:
            messages.info(request, 'Este modelo ya estaba guardado.')
            return _dashboard_redirect('results-pane', results_view='dbscan')
        form = DBSCANSaveForm(request.POST)
        if not form.is_valid():
            messages.error(request, 'Indica un nombre válido para guardar el modelo.')
            return _dashboard_redirect('results-pane', results_view='dbscan')
        run.name = form.cleaned_data['name']
        run.topic = form.cleaned_data['topic']
        run.description = form.cleaned_data['description']
        run.is_saved = True
        run.saved_at = timezone.now()
        run.save(update_fields=(
            'name', 'topic', 'description', 'is_saved', 'saved_at',
        ))
        messages.success(request, 'El modelo se guardó en el catálogo.')
        return _dashboard_redirect('results-pane', results_view='dbscan')


class DBSCANDeleteView(View):
    """Delete a single saved DBSCANRun."""

    def post(self, request, pk):
        DBSCANRun.objects.filter(pk=pk).delete()
        return _dashboard_redirect('models-pane')

