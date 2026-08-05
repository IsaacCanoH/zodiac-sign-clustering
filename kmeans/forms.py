from django import forms


class KMeansTrainingForm(forms.Form):
    name = forms.CharField(max_length=150, required=False)
    topic = forms.CharField(max_length=150, required=False)
    description = forms.CharField(required=False)
    algorithm = forms.ChoiceField(
        choices=(('kmeans', 'K-Means'),),
        initial='kmeans',
    )
    cluster_count = forms.IntegerField(min_value=2)
    columns = forms.MultipleChoiceField(
        error_messages={
            'required': (
                'Selecciona al menos una columna numérica en la sección '
                '“Columnas para el entrenamiento”.'
            ),
            'invalid_choice': 'Una de las columnas seleccionadas ya no está disponible.',
        },
    )
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
        self.fields['cluster_count'].max_value = max_clusters
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
        comparison_column = cleaned_data.get('comparison_column')
        columns = cleaned_data.get('columns', [])
        if comparison_column and comparison_column in columns:
            self.add_error(
                'columns',
                'La columna de comparación no puede utilizarse para entrenar.',
            )
        return cleaned_data


class KMeansSaveForm(forms.Form):
    name = forms.CharField(max_length=150)
    topic = forms.CharField(max_length=150, required=False)
    description = forms.CharField(required=False)
