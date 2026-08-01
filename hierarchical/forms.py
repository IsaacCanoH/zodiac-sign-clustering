from django import forms


class HierarchicalTrainingForm(forms.Form):
    name = forms.CharField(max_length=150, required=False)
    topic = forms.CharField(max_length=150, required=False)
    description = forms.CharField(required=False)
    n_clusters = forms.IntegerField(min_value=2, label='Número de clusters')
    linkage = forms.ChoiceField(
        choices=[
            ('ward', 'Ward (Minimiza la varianza)'),
            ('complete', 'Complete (Máxima distancia)'),
            ('average', 'Average (Distancia promedio)'),
            ('single', 'Single (Mínima distancia)'),
        ],
        initial='ward',
        label='Método de enlace',
    )
    affinity = forms.ChoiceField(
        choices=[
            ('euclidean', 'Euclidiana'),
            ('manhattan', 'Manhattan'),
            ('cosine', 'Coseno'),
        ],
        initial='euclidean',
        label='Métrica de distancia',
    )
    scaling_method = forms.ChoiceField(
        choices=[
            ('standard', 'Estandarización (Z-score)'),
            ('minmax', 'Min-Max (0 a 1)'),
            ('none', 'Sin escalar'),
        ],
        initial='standard',
        label='Escalado',
    )
    columns = forms.MultipleChoiceField()
    comparison_column = forms.ChoiceField(required=False)

    def __init__(
        self,
        *args,
        numeric_columns=(),
        categorical_columns=(),
        max_clusters=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fields['n_clusters'].max_value = max_clusters
        self.fields['columns'].choices = [
            (column['name'], column['name']) for column in numeric_columns
        ]
        self.fields['comparison_column'].choices = [
            ('', 'Sin comparación'),
            *[
                (column['name'], column['name'])
                for column in categorical_columns
            ],
        ]

    def clean_columns(self):
        columns = self.cleaned_data['columns']
        return list(dict.fromkeys(columns))

    def clean(self):
        cleaned_data = super().clean()
        linkage = cleaned_data.get('linkage')
        affinity = cleaned_data.get('affinity')
        comparison_column = cleaned_data.get('comparison_column')
        columns = cleaned_data.get('columns', [])

        # Ward only works with euclidean distance
        if linkage == 'ward' and affinity != 'euclidean':
            self.add_error(
                'affinity',
                'El método de enlace Ward requiere usar la distancia Euclidiana.',
            )

        if comparison_column and comparison_column in columns:
            self.add_error(
                'columns',
                'La columna de comparación no puede utilizarse para entrenar.',
            )
        return cleaned_data


class HierarchicalSaveForm(forms.Form):
    name = forms.CharField(max_length=150)
    topic = forms.CharField(max_length=150, required=False)
    description = forms.CharField(required=False)