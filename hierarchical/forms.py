from django import forms
from .models import HierarchicalRun

class HierarchicalTrainingForm(forms.ModelForm):
    class Meta:
        model = HierarchicalRun
        fields = ['n_clusters', 'linkage', 'affinity', 'scaling_method']
        widgets = {
            'n_clusters': forms.NumberInput(attrs={'class': 'form-control', 'min': 2, 'max': 20}),
            'linkage': forms.Select(
                choices=[
                    ('ward', 'Ward (Minimiza la varianza)'),
                    ('complete', 'Complete (Máxima distancia)'),
                    ('average', 'Average (Distancia promedio)'),
                    ('single', 'Single (Mínima distancia)')
                ], 
                attrs={'class': 'form-control'}
            ),
            'affinity': forms.Select(
                choices=[
                    ('euclidean', 'Euclidiana'),
                    ('manhattan', 'Manhattan'),
                    ('cosine', 'Coseno')
                ],
                attrs={'class': 'form-control'}
            ),
            'scaling_method': forms.Select(
                choices=[
                    ('standard', 'Estandarización (Z-score)'),
                    ('minmax', 'Min-Max (0 a 1)'),
                    ('none', 'Sin escalar')
                ],
                attrs={'class': 'form-control'}
            )
        }

    def clean(self):
        cleaned_data = super().clean()
        linkage = cleaned_data.get('linkage')
        affinity = cleaned_data.get('affinity')

        # Regla matemática de scikit-learn: El enlace 'ward' SOLO funciona con distancia 'euclidean'
        if linkage == 'ward' and affinity != 'euclidean':
            self.add_error('affinity', 'El método de enlace Ward requiere usar la distancia Euclidiana.')
            
        return cleaned_data