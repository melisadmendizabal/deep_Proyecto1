# Proyecto 1 — Competencia de Modelación (Deep Learning y Sistemas Inteligentes)

Predicción de SalePrice (precio de venta de casas) con un MLP en PyTorch. Todo el trabajo — EDA, preprocesamiento, definición de arquitecturas, búsqueda de hiperparámetros, selección del modelo ganador y generación de las entregas de la competencia vive en proyecto.ipynb.

## Estructura del proyecto

```
.
├── proyecto.ipynb              # notebook con TODO el pipeline (única fuente de código)
├── train.csv                   # dataset de entrenamiento (Kaggle House Prices)
├── resultados_modelos.csv      # log acumulado de cada modelo entrenado (se genera/actualiza solo)
├── modelos/                    # checkpoints .pt de cada modelo entrenado (se genera solo)
│   ├── modelo_mlp_bn.pt
│   ├── modelo_mlp_sinbn.pt
│   ├── modelo_mlp_lbfgs.pt
│   ├── modelo_mlp_linearskip.pt
│   ├── modelo_mlp_sklearn.pt
│   ├── grid_*.pt, greedy_*.pt          # checkpoints de cada combinación probada
│   └── modelo_ganador_*.pt             # modelo final exportado (el que se usa para predecir)
├── prueba/
│   ├── pipeline_test.csv       # input para probar el formato de entrega
│   └── resultado_test.csv      # output generado por el notebook
└── competencia/
    ├── test_features-1.csv     # input real de la competencia
    └── resultado_competencia.csv  # output/entrega generado por el notebook
```

> modelos/ y resultados_modelos.csv no hay que crearlos a mano: el notebook los genera y los va llenando la primera vez que corre.

## Requisitos

- Python 3.11 (la versión con la que se guardó el notebook; 3.9+ debería funcionar igual)
- Jupyter (Notebook o JupyterLab)
- Paquetes:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

No se usa GPU de forma obligatoria: el notebook detecta automáticamente `cuda` si está disponible y si no, usa `cpu` (los tiempos de entrenamiento en CPU son razonables para este tamaño de dataset, ~1460 filas).

## Cómo reproducir los resultados

1. **Colocar los archivos de datos** en la misma carpeta que `proyecto.ipynb`, respetando la estructura de arriba:
   - `train.csv` en la raíz
   - `prueba/pipeline_test.csv`
   - `competencia/test_features-1.csv`

2. **Abrir `proyecto.ipynb`** y correrlo **de arriba hacia abajo, en orden** (`Run All`). El notebook no está pensado para correr celdas sueltas: las variables (`df_clean`, `X`, `scaler`, `train_loader`, etc.) se definen progresivamente y las celdas de más abajo dependen de las de arriba. La única excepción es la celda que arma `predecir_dataset` / `generar_submission` (la de "Exportar el modelo final" en adelante), que sí se redefine todo lo que necesita para poder correr de forma independiente.

3. **Qué hace cada bloque, en orden:**
   - *Análisis exploratorio*: carga `train.csv`, describe variables, nulos, outliers.
   - *Preprocesamiento*: imputación de nulos, encoding ordinal y one-hot, remoción de outliers de `GrLivArea`, transformación `log1p` del target.
   - *Split*: separa train/val/test (70/15/15) y escala con `StandardScaler` (fit solo con train).
   - *Definición de arquitecturas*: `MLP` (con BatchNorm), `MLP_sinBN`, `MLP_LinearSkip`, `MLP_LBFGS`.
   - *Búsqueda de hiperparámetros* (4 rondas: grid search con 3 semillas, grid ampliado, grid con K-Fold, greedy search, y una variante LBFGS). Cada combinación probada queda registrada en `resultados_modelos.csv` y su checkpoint en `modelos/`.
   - *Selección del modelo ganador*: se compara todo lo anterior contra una regresión lineal de referencia y se exporta el mejor modelo como `modelos/modelo_ganador_*.pt`.
   - *Análisis de residuos* del modelo final.
   - *Generación de entregas*: usa `generar_submission(...)` para producir `prueba/resultado_test.csv` y `competencia/resultado_competencia.csv`.

4. **Entrenamiento incremental (importante):** el notebook está diseñado para **no reentrenar de cero cada vez que se corre**. Antes de entrenar cualquier bloque (grid, greedy, o el modelo final), revisa si ya existe una fila con ese nombre en `resultados_modelos.csv` y, si el `.pt` correspondiente sigue existiendo en `modelos/`, simplemente lo carga. Esto significa:
   - La primera corrida completa sí entrena todo (puede tardar varios minutos, dependiendo del hardware).
   - Corridas posteriores son casi instantáneas porque reutilizan los checkpoints ya guardados.
   - **Para forzar un reentrenamiento total desde cero**, borra `resultados_modelos.csv` y la carpeta `modelos/` antes de correr el notebook.

## Nota sobre las celdas de generación de entregas

Las dos últimas celdas del notebook (predicción sobre `prueba/` y sobre `competencia/`) buscan el `modelo_ganador_*.pt` más reciente, pero **actualmente el `path_modelo` que le pasan a `generar_submission(...)` está fijo en `modelos/modelo_mlp_sinbn.pt`**, no en el ganador detectado. Si quieres que la entrega final use realmente el modelo seleccionado como ganador (sección "Exportación del modelo ganador"), cambia ese argumento por `path_modelo` (la variable que ya calcula `max(candidatos_ganador, key=os.path.getmtime)` unas líneas arriba), o apunta directo al archivo `modelo_ganador_*.pt` que te interese.

## Salidas esperadas

Al terminar de correr el notebook deberías tener:
- `resultados_modelos.csv` con todas las combinaciones evaluadas y su `val_rmse_log`.
- `modelos/modelo_ganador_*.pt` — el modelo final.
- `prueba/resultado_test.csv` y `competencia/resultado_competencia.csv` — archivos de predicción con columnas `Id, Prediction`.
