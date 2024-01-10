# FlaskTesina
Hace falta tener instalado miniconda y una GPU de Nvidia (en caso contrario, deberia de funcionar utilizando la CPU de todas formas, pero no fue testeado desde los ultimos cambios y cuando fue testeado resulta ser mucho mas lento)

#Miniconda

para crear el entorno utilizar
```
conda env create --name envname --file=environments.yml
```
o utilizar ```--prefix PATH``` en vez de ```--name envname``` para que el entorno se cree en el ```PATH``` especificado

para iniciar el entorno utilizar
```
conda activate envname
```
o en el caso de crear el entorno en un ```PATH``` indicado manualmente, utilizar
```
conda activate PATH
```
ya que al no estar instalado en el sitio por defecto, conda no encontraria el entorno solo por su nombre

de manera similar, para salir del entorno utilizar
```
conda deactivate
```

para ejecutar el programa, dentro del entorno utilizar
```
python app.py
```

con lo que es posible acceder a 127.0.0.1:5000 (localhost en el puerto 5000) para tener una interfaz grafica y poder analizar las imagenes.

En la consola se muestra informacion como la clasificacion en si o el tiempo restante
