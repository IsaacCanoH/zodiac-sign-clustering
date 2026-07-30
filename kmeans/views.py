from django.shortcuts import redirect
from django.urls import reverse
from django.views import View

from datasets.models import Dataset

from .forms import KMeansTrainingForm
from .services import (
    KMeansTrainingError,
    build_training_setup,
    clear_kmeans_runs,
    train_kmeans,
)


class KMeansTrainingView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if not dataset:
            return redirect(f"{reverse('dashboard:index')}#training-pane")

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
            return redirect(f"{reverse('dashboard:index')}#training-pane")

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
            return redirect(f"{reverse('dashboard:index')}#training-pane")

        return redirect(f"{reverse('dashboard:index')}#results-pane")


class KMeansResetView(View):
    def post(self, request):
        dataset = Dataset.objects.filter(pk=1).first()
        if dataset:
            clear_kmeans_runs(dataset)
        request.session.pop('kmeans_form_state', None)
        return redirect(f"{reverse('dashboard:index')}#training-pane")
