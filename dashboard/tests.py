from django.test import TestCase
from django.urls import reverse


class DashboardViewTests(TestCase):
    def test_dashboard_is_the_home_page(self):
        response = self.client.get(reverse('dashboard:index'))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'dashboard/index.html')
        self.assertContains(response, 'Datos cargados')

    def test_dashboard_does_not_accept_post_requests(self):
        response = self.client.post(reverse('dashboard:index'))

        self.assertEqual(response.status_code, 405)
