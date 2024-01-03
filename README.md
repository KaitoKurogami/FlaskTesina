# FlaskTesina
Hace falta tener instalado miniconda y una GPU de Nvidia (en caso contrario, deberia de funcionar utilizando la CPU de todas formas, pero no fue testeado desde los ultimos cambios y cuando fue testeado resulta ser mucho mas lento)

#Miniconda

para crear el entorno utilizar
'conda env create --name envname --file=environments.yml'

para iniciar el entorno utilizar
'conda activate ./envname'

para ejecutar el programa
'python app.py'

con lo que es posible acceder a 127.0.0.1:5000 (localhost en el puerto 5000) para tener una interfaz grafica y poder analizar las imagenes
en la consola se muestra informacion como la clasificacion en si o el tiempo restante
