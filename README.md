# Implementación del optimizador ADAM (mini-batch) en Regresión Lineal

Notebook: `ADAM_MiniBatch.ipynb`

Se implementa **desde cero** el optimizador **ADAM** con **mini-batch** para ajustar una regresión lineal simple

```
y = b0 + b1 · x
```

y se compara contra:

1. La **solución cerrada** de scikit-learn (`LinearRegression`).
2. El **Adam de scikit-learn** (`MLPRegressor`) con los mismos hiperparámetros.

## Datos

30 puntos con una relación lineal casi perfecta (`x = 1..30`, `y` de 5 a 55). La correlación entre `x` e `y` se comprueba con `df.corr()`.

## Contenido del notebook

| Sección | Descripción |
|---|---|
| Librerías y datos | `pandas`, `numpy`, `matplotlib`; datos y análisis de correlación |
| **ADAM** | Función `ADAM(x, y, ln, epoch, batch, beta1, beta2)` implementada con NumPy |
| **Comparación** | `LinearRegression` y `MLPRegressor` (Adam de sklearn) |
| **Tabla comparativa** | `b0`, `b1` y MSE de los tres métodos |

## Cómo funciona la implementación

En cada época:

1. Se barajan los índices de los datos (`np.random.permutation`).
2. Se recorren los datos en lotes de tamaño `batch`. El último lote puede ser más pequeño (`min(i + batch, n)`).
3. Para cada lote se calculan los gradientes del MSE:
   - `db0 = Σ -(2/m)(y - ŷ)`
   - `db1 = Σ -(2/m) · x · (y - ŷ)`
4. Se actualizan los momentos de primer y segundo orden:
   - `m = β1·m + (1-β1)·g`
   - `v = β2·v + (1-β2)·g²`
5. Se aplica la corrección de sesgo (`m̂ = m / (1-β1^t)`, `v̂ = v / (1-β2^t)`).
6. Se actualizan los parámetros: `b = b - lr · m̂ / (√v̂ + ε)`.

Los parámetros parten de `b0 = b1 = 0` y `ε = 1e-8`.

### Hiperparámetros usados

| Parámetro | Valor |
|---|---|
| Learning rate | `0.00005` |
| Épocas | `50000` |
| Batch size | `5` |
| β1 | `0.9` |
| β2 | `0.999` |

## Comparación con sklearn

### Solución cerrada

```python
from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(x.reshape(-1, 1), y)
```

### Adam de sklearn

sklearn no ofrece Adam como optimizador suelto, solo dentro de sus redes neuronales. Una `MLPRegressor` **sin capas ocultas** y con activación `identity` equivale a `y = b0 + b1·x`:

```python
mlp = MLPRegressor(hidden_layer_sizes=(), activation="identity", solver="adam",
                   batch_size=5, learning_rate_init=0.00005, beta_1=0.9, beta_2=0.999,
                   max_iter=50000, tol=0, n_iter_no_change=10**9, alpha=0.0)
```

Ajustes necesarios para que sea comparable con la implementación propia:

- `tol=0` y `n_iter_no_change=10**9`: desactivan la parada temprana, de modo que entrena las 50.000 épocas completas.
- `alpha=0.0`: elimina la regularización L2 (la pérdida es solo el MSE).

## Resultados

| Método | b0 | b1 | MSE | RMSE | R² |
|---|---|---|---|---|---|
| `LinearRegression` (solución cerrada) | 3.26897 | 1.71813 | 0.53787 | 0.73340 | 0.99757 |
| ADAM desde cero | 3.26879 | 1.71810 | 0.53787 | 0.73340 | 0.99757 |
| Adam de sklearn (`MLPRegressor`) | 3.26889 | 1.71812 | 0.53787 | 0.73340 | 0.99757 |

Valores obtenidos al ejecutar la comparación con los hiperparámetros anteriores. Pueden variar ligeramente por el orden aleatorio de los mini-batches y la inicialización del `MLPRegressor`.

## Conclusiones

- La implementación de ADAM converge a la misma recta que la solución cerrada; las diferencias aparecen a partir de la 4.ª o 5.ª cifra decimal.
- El resultado coincide con el Adam de sklearn, lo que valida la implementación.
- Con `lr = 0.00005` hacen falta unos 300.000 pasos. Con `lr = 0.01` bastan unas 3.000 épocas para llegar al mismo resultado.
- En un problema tan pequeño (30 puntos, 2 parámetros) la solución cerrada es exacta e instantánea. Adam se justifica cuando hay muchos datos o parámetros, o cuando no existe solución cerrada.

## Requisitos

```
numpy
pandas
matplotlib
scikit-learn
```

## Ejecución

Abrir `ADAM_MiniBatch.ipynb` en Google Colab o Jupyter y ejecutar las celdas en orden.
