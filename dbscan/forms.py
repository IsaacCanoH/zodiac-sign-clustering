from django import forms


class DBSCANTrainingForm(forms.Form):
    name = forms.CharField(max_length=150, required=False)
    topic = forms.CharField(max_length=150, required=False)
    description = forms.CharField(required=False)
    epsilon = forms.FloatField(
        min_value=0.01,
        label='Epsilon (ε)',
    )
    min_samples = forms.IntegerField(
        min_value=2,
        label='Mínimo de muestras',
    )
    columns = forms.MultipleChoiceField()
    comparison_column = forms.ChoiceField(required=False)

    def __init__(
        self,
        *args,
        numeric_columns=(),
        categorical_columns=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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


class DBSCANSaveForm(forms.Form):
    name = forms.CharField(max_length=150)
    topic = forms.CharField(max_length=150, required=False)
    description = forms.CharField(required=False)
