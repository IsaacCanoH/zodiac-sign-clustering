from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from datasets.models import Dataset
from .models import HierarchicalRun
from .services import train_hierarchical_model
from django.urls import reverse

def workspace(request):
    return redirect('dashboard:index')

def train(request):
    if request.method == 'POST':
        dataset = Dataset.objects.last()
        
        n_clusters = int(request.POST.get('n_clusters', 3))
        linkage = request.POST.get('linkage', 'ward')
        affinity = request.POST.get('affinity', 'euclidean')
        comparison_column = request.POST.get('comparison_column', '')
        selected_columns = request.POST.getlist('selected_columns') 
        
        if not selected_columns:
            messages.error(request, 'Debes seleccionar al menos una columna numérica para entrenar el algoritmo.')
            return redirect('dashboard:index')
            
        if linkage == 'ward' and affinity != 'euclidean':
            messages.error(request, 'El método Ward requiere obligatoriamente el uso de distancia Euclidiana.')
            return redirect('dashboard:index')

        run_record, labels = train_hierarchical_model(
            dataset_instance=dataset,
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity,
            scaling_method='standard',
            selected_columns=selected_columns,
            comparison_column=comparison_column
        )
        
        # Guardar en sesión como activo y seleccionar la pestaña de resultados/algoritmo
        request.session['active_hierarchical_run'] = run_record.id
        request.session['active_algorithm'] = 'hierarchical'
        
        messages.success(request, 'Clustering Jerárquico ejecutado con éxito.')
        return redirect(reverse('dashboard:index') + '#results-pane')
            
    return redirect(reverse('dashboard:index') + '#results-pane')

def activate(request, run_id):
    """Activa un modelo guardado para mostrarlo en la pestaña Resultados."""
    if request.method == 'POST':
        run = get_object_or_404(HierarchicalRun, id=run_id)
        request.session['active_hierarchical_run'] = run.id
        request.session['active_algorithm'] = 'hierarchical'
        messages.success(request, 'Modelo de Clustering Jerárquico activado correctamente.')
    return redirect(reverse('dashboard:index') + '#results-pane')

def delete(request, run_id):
    """Elimina un modelo guardado del historial."""
    if request.method == 'POST':
        run = get_object_or_404(HierarchicalRun, id=run_id)
        if request.session.get('active_hierarchical_run') == run.id:
            request.session.pop('active_hierarchical_run', None)
        run.delete()
        messages.success(request, 'Modelo de Clustering Jerárquico eliminado.')
    return redirect(reverse('dashboard:index') + '#results-pane')

def results(request, run_id):
    run_record = get_object_or_404(HierarchicalRun, id=run_id)
    return render(request, 'hierarchical/results_workspace.html', {
        'run': run_record,
        'dataset': run_record.dataset
    })