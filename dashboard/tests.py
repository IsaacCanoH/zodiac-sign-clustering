from django.test import TestCase
from django.urls import reverse

from datasets.models import Dataset
from dbscan.services import train_dbscan
from kmeans.services import train_kmeans


class DashboardViewTests(TestCase):
    def test_dashboard_is_the_home_page(self):
        response = self.client.get(reverse('dashboard:index'))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'dashboard/index.html')
        self.assertContains(response, 'Datos cargados')

    def test_dashboard_does_not_accept_post_requests(self):
        response = self.client.post(reverse('dashboard:index'))

        self.assertEqual(response.status_code, 405)

    def test_saved_models_use_four_column_scrollable_card_grid(self):
        dataset = Dataset.objects.create(
            pk=1,
            source_name='modelos.csv',
            columns=['x', 'y'],
            records=[
                {'x': '0', 'y': '0'},
                {'x': '0.1', 'y': '0.1'},
                {'x': '5', 'y': '5'},
                {'x': '5.1', 'y': '5.1'},
            ],
        )
        train_kmeans(dataset, ['x', 'y'], 2)
        train_dbscan(dataset, ['x', 'y'], epsilon=0.1, min_samples=2)

        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(
            response,
            'max-height: 35rem; overflow-y: auto; overscroll-behavior: contain;',
        )
        self.assertContains(
            response,
            'class="col-sm-6 col-lg-4 col-xl-3"',
            count=2,
        )
        self.assertContains(response, 'Todos los modelos')
        self.assertContains(response, 'K-Means')
        self.assertContains(response, 'DBSCAN')
        self.assertEqual(
            {item['algorithm'] for item in response.context['all_saved_models']},
            {'kmeans', 'dbscan'},
        )
