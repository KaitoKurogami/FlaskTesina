# FlaskTesina
Hace falta tener instalado miniconda y una GPU de Nvidia (en caso contrario, deberia de funcionar utilizando la CPU de todas formas, pero no fue testeado desde los ultimos cambios y cuando fue testeado resulta ser mucho mas lento)

## Miniconda

para crear el entorno utilizar
```
%USERPROFILE%\miniconda3\condabin\conda env create --name envname --file=environment.yml
```
o utilizar ```--prefix FILEPATH``` en vez de ```--name envname``` para que el entorno se cree en el ```FILEPATH``` especificado

para iniciar el entorno utilizar
```
%USERPROFILE%\miniconda3\condabin\conda activate envname
```
o en el caso de crear el entorno en un ```FILEPATH``` indicado manualmente, utilizar
```
%USERPROFILE%\miniconda3\condabin\conda activate FILEPATH
```
ya que al no estar instalado en el sitio por defecto, conda no encontraria el entorno solo por su nombre
En caso de haber configurado durante la instalacion de miniconda para agregar miniconda a la variable de entorno PATH, es posible reemplazar ```%USERPROFILE%\miniconda3\condabin\conda``` por simplemente ```conda```


de manera similar, para salir del entorno utilizar
```
conda deactivate
```

para ejecutar el programa, dentro del entorno utilizar:
```
python app.py
```

con lo que es posible acceder a 127.0.0.1:5000 (localhost en el puerto 5000) para tener una interfaz grafica y poder analizar las imagenes.

## Funcionamiento

En la consola se muestra informacion como la clasificacion en si, o el tiempo restante.

las imagenes resultantes son guardadas en la carpeta  ```static/files/result```

Se puede elegir la red neuronal a utiilizar entre tres opciones: VGG16, ResNet50, o una Red propia.

En caso de utilizar una red propia, se deben respetar ciertas restricciones:
```
La red debe encontrarse como un archivo de formato hdf5 (archivo con extension .h5) en la carpeta "static/files/models".

La red tiene que haber sido creada mediante transfer learning en base a una red VGG16 (actualmente se aceptan
redes realizadas mediante el método de transfer learning provisto por Keras, por lo que se deben respetar
formatos y caracteristicas, tanto de salida como de entrada, que poseen dichas redes). Además
se deben ingresar las categorias por las cuales la red clasifica imagenes, separadas por comas.

Los resultados de estas redes pueden analizarse mediante los métodos LIME y SHAP, ya que la implementación del método
Grad-CAM no fue adaptada en su totalidad todavia.
```

Respecto a los formularios

Formulario del método SHAP
```
	Este formulario permite modificar los parámetros en caso de querer utilizar el método SHAP.

	El campo de “Evaluations” permite indicar la cantidad de evaluaciones, lo que se traduce en la cantidad de permutaciones de contribuciones a analizar, por lo que mientras mayor sea este numero, mayor cantidad de análisis realizados, mayor granularidad, pero también un mayor tiempo necesario para el procesamiento de la información

	El campo de Batch Size representa la cantidad de muestras utilizadas en cada pasada, a mayor batch size, mayor velocidad, pero también hace falta utilizar más memoria, y suele implicar una disminución en la precisión
 ```

Formulario del método LIME
```
	Este formulario permite modificar los parámetros en caso de querer utilizar el método LIME.

	Perturbations representa la cantidad de permutaciones a analizar, siendo estas las combinaciones posibles de superpixeles habilitados y deshabilitados.

	Kernel size es un parámetro que afecta a la generación de superpixeles en sí. La función que los genera utiliza una distribución gaussiana, y el kernel size se corresponde con la desviación estándar, por lo que a mayor número, mayor tamaño de los superpixeles y menor cantidad.

	Maximum distance of perturbation también está relacionado con la generación de superpixeles, indicando un punto de corte para las distancias de los datos, por lo que un valor mayor implica un tamaño mayor de los superpixeles y una menor cantidad.

	Ratio of perturbations equilibra la proximidad del espacio de color y la proximidad del espacio de imagen. Los valores más altos dan más peso al espacio de color. Es decir, altera la forma de los superpixeles generados, causando que estos se guien mas por las diferencias de color en la imagen.
```

Formulario del método Grad-CAM
```
	Este formulario no posee campos debido a como es el funcionamiento del método Grad-CAM en sí, como este realiza la media aritmética de los gradientes de salida para una clase con respecto a cada mapa de activaciones de una capa concreta, el procedimiento siempre es el mismo.
```
