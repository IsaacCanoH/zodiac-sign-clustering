from django.test import TestCase
from django.urls import reverse

from datasets.models import Dataset


class HierarchicalResultsViewTests(TestCase):
    def test_hierarchical_training_opens_complete_results_panel(self):
        Dataset.objects.create(
            pk=1,
            source_name='puntos.csv',
            columns=['x', 'y'],
            records=[
                {'x': '0', 'y': '0'},
                {'x': '0.1', 'y': '0.2'},
                {'x': '10', 'y': '10'},
                {'x': '10.1', 'y': '10.2'},
            ],
        )

        response = self.client.post(
            reverse('hierarchical:train'),
            {
                'algorithm': 'hierarchical',
                'n_clusters': 2,
                'linkage': 'ward',
                'affinity': 'euclidean',
                'scaling_method': 'standard',
                'columns': ['x', 'y'],
                'comparison_column': '',
            },
        )

        self.assertRedirects(
            response,
            f"{reverse('dashboard:index')}?results_view=hierarchical#results-pane",
        )
        results = self.client.get(
            reverse('dashboard:index'),
            {'results_view': 'hierarchical'},
        )
        self.assertEqual(results.context['ui_active_algorithm'], 'hierarchical')
        self.assertContains(
            results,
            'id="hierarchical-results-panel" role="tabpanel"',
        )
        self.assertContains(results, 'hierarchicalClusterChart')
