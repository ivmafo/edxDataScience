import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# casa idx
X = np.array([[1],[2],[3]]) # variables de caracteristicas (aqui es tamaño de la casa)

# precio casa
Y = np.array([[50000],[100000],[150000]])

# prediccion del modelo 
knn = KNeighborsRegressor(2)

# entrenamiento del modelo
knn.fit(X, Y)

# Predecir el precio de una nueva casa
nuevo_punto = np.array([[10]])
prediccion = knn.predict(nuevo_punto)

# Graficar los datos de entrenamiento
plt.scatter(X, Y, color='blue', label='Datos de entrenamiento')

# Graficar el nuevo punto
plt.scatter(nuevo_punto, prediccion, color='red', label='Predicción')

# Añadir etiquetas a los ejes
plt.xlabel('Tamaño de la casa')
plt.ylabel('Precio de la casa')

# Añadir título al gráfico
plt.title('KNeighborsRegressor: Predicción del Precio de la Casa')


# Añadir leyenda
plt.legend()

# Mostrar el gráfico
plt.show()