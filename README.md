# Comandos básicos de Django

## Levantar el proyecto

```bash
python manage.py runserver
```

## Crear un módulo o aplicación

```bash
python manage.py startapp nombre_modulo
```

## Crear migraciones

```bash
python manage.py makemigrations nombre_modulo
```

## Aplicar migraciones

```bash
python manage.py migrate
```

## Crear un usuario administrador

```bash
python manage.py createsuperuser
```

# Estructura de cada módulo

```text
nombre_modulo/
├── migrations/
│   └── __init__.py
├── templates/
├── __init__.py
├── admin.py
├── apps.py
├── models.py
├── tests.py
├── urls.py
└── views.py
```