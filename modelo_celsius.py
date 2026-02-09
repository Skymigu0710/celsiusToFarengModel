import tensorflow as tf
import numpy as np

# Datos
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float).reshape(-1,1)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float).reshape(-1,1)

# Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=[1]),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1)
])

modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento
modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)

# Guardar modelo en formato TensorFlow
modelo.save('modelo_celsius.h5')
print("Modelo guardado ✅")
