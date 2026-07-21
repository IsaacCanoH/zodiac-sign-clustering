import csv
import io

from django import forms


MAX_FILE_SIZE = 10 * 1024 * 1024


class DatasetUploadForm(forms.Form):
    file = forms.FileField(
        label='Archivo de datos',
        widget=forms.ClearableFileInput(
            attrs={'class': 'form-control', 'accept': '.csv'}
        ),
    )

    def clean_file(self):
        uploaded_file = self.cleaned_data['file']

        if not uploaded_file.name.lower().endswith('.csv'):
            raise forms.ValidationError('Selecciona un archivo con formato CSV.')
        if len(uploaded_file.name) > 255:
            raise forms.ValidationError('El nombre del archivo es demasiado largo.')
        if uploaded_file.size > MAX_FILE_SIZE:
            raise forms.ValidationError('El archivo no debe superar los 10 MB.')

        return uploaded_file

    def clean(self):
        cleaned_data = super().clean()
        uploaded_file = cleaned_data.get('file')
        if not uploaded_file:
            return cleaned_data

        try:
            content = uploaded_file.read().decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise forms.ValidationError(
                'El archivo debe utilizar codificación UTF-8.'
            ) from error

        try:
            dialect = csv.Sniffer().sniff(content[:4096], delimiters=',;\t|')
        except csv.Error:
            dialect = csv.excel

        try:
            reader = csv.DictReader(io.StringIO(content), dialect=dialect)
            columns = [column.strip() for column in (reader.fieldnames or [])]
        except csv.Error as error:
            raise forms.ValidationError(
                'No fue posible interpretar el archivo CSV.'
            ) from error

        if not columns or any(not column for column in columns):
            raise forms.ValidationError('El archivo debe incluir encabezados válidos.')
        if len(columns) != len(set(columns)):
            raise forms.ValidationError('Los nombres de las columnas no pueden repetirse.')

        records = []
        for row in reader:
            if None in row:
                raise forms.ValidationError(
                    'Hay registros con más valores que columnas.'
                )
            normalized = {
                columns[index]: (row[original] or '').strip()
                for index, original in enumerate(reader.fieldnames)
            }
            if any(normalized.values()):
                records.append(normalized)

        if not records:
            raise forms.ValidationError('El archivo no contiene registros de datos.')

        cleaned_data['columns'] = columns
        cleaned_data['records'] = records
        return cleaned_data
