import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importamos las herramientas nativas del proyecto
from kmeans.services import _filtered_rows, _build_matrix, _comparison_summary
from .models import HierarchicalRun

def train_hierarchical_model(dataset_instance, n_clusters, linkage, affinity, scaling_method, selected_columns, comparison_column):
    category_filter, rows = _filtered_rows(dataset_instance, None)
    
    if len(rows) == 0:
        raise ValueError("El dataset no contiene registros válidos para entrenar.")
    if n_clusters >= len(rows):
        raise ValueError("La cantidad de grupos debe ser menor que la cantidad de registros.")
        
    matrix, imputed_values = _build_matrix(rows, selected_columns)

    if len(np.unique(matrix, axis=0)) < n_clusters:
        raise ValueError(f'No existen suficientes combinaciones para {n_clusters} grupos.')

    if scaling_method == 'standard':
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(matrix)
    else:
        X_processed = matrix

    model = AgglomerativeClustering(n_clusters=n_clusters, metric=affinity, linkage=linkage)
    raw_labels = model.fit_predict(X_processed)
    
    labels = [int(label) + 1 for label in raw_labels]
    
    score = float(silhouette_score(X_processed, raw_labels, metric=affinity)) if len(set(raw_labels)) > 1 else None

    assignments = []
    for (row_number, _), cluster in zip(rows, labels, strict=True):
        assignments.append({'row_number': row_number, 'cluster': cluster})

    cluster_sizes = {str(c): labels.count(c) for c in range(1, n_clusters + 1)}

    comparison = (
        _comparison_summary(rows, labels, comparison_column, n_clusters)
        if comparison_column else {
            'values': [], 'summaries': [], 'valid_count': 0, 'overall_match_percentage': None,
        }
    )

    run_record = HierarchicalRun.objects.create(
        dataset=dataset_instance,
        n_clusters=n_clusters,
        linkage=linkage,
        affinity=affinity,
        scaling_method=scaling_method,
        selected_columns=selected_columns,
        assignments=assignments,
        cluster_sizes=cluster_sizes,
        comparison_column=comparison_column,
        comparison_values=comparison['values'],
        cluster_comparison=comparison['summaries'],
        comparison_valid_count=comparison['valid_count'],
        overall_match_percentage=comparison['overall_match_percentage'],
        cluster_count=n_clusters,
        sample_count=len(rows),
        silhouette_score=score
    )
    
    return run_record, labels