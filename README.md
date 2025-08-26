# Multietiqueta de Literatura Médica

Clasificador multietiqueta para artículos biomédicos (título + abstract) en 4 dominios: **cardio**, **hepato‑renal**, **neuro** y **onco**.

Resumen rápido:
- Backbone: PubMedBERT (Hugging Face)
- Cabeza: Label‑Wise Attention
- Umbrales por clase optimizados con sweep de F1.5 (prioriza recall)
- Soporta encoder local en `src/model/local_pubmedbert`

## Contenido y estructura

Principales archivos y carpetas:

```
train.py                 # Entrenamiento (script principal)
run_eval.py             # Evaluación rápida desde CSV
app.py                  # Streamlit UI
src/                    # Código fuente (data, model, infer, train, eval, utils)
data/                   # Ejemplos de CSV / splits
outputs/                # Carpeta por corrida con artefactos (final_model.pt, thresholds.npy, labels.json)
```

Modelo — ejemplo de outputs por corrida:

```
outputs/
   ├─ b3/
   │   ├─ best_model.pt
   │   ├─ final_model.pt
   │   ├─ thresholds.npy
   │   └─ labels.json
```
## Uso

Para utilizar esta solución, se debe:
- Clonar el repositorio (tomará cierto tiempo):
```powershell
git clone https://github.com/acarmonag/Medical-Literature-Classifier
```
- Al terminar el clone, abrir un terminal dentro de la carpeta Medical-Literature-Classifier y ejecutar estas líneas
```powershell
git lfs install
```
```powershell
git lfs pull
```
- Ahora, se deben instalar las dependencias: Unix/macOS:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
- O en Windows:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
- En la carpeta Medical-Literature-Classifier, abrir una terminal y ejecutar:
```bash
streamlit run app.py
```
- Si al ejecutar el comando anterior la terminal solicita un email, solo es necesario darle enter.

## Disclaimers de la UI (IMPORTANTE)
- Lo anterior, debería abrir una interfaz de usuario donde se puede cargar un archivo csv con diferentes artículos, o un simple artículo con título y abstract, al llenar los campos necesarios, el programa ejecutará automáticamente la clasificación.
- Si se sube un archivo csv, el programa iniciará la clasificación de manera inmediata. NO ES NECESARIO PRESIONAR EL BOTÓN PREDECIR.
- Al terminar la clasificación por .csv, parecerá que se hubiera recargado, pero solo se tiene que volver a la opción predicción por CSV + Métricas.
- !!! Si sale un error al momento de ejecutar la UI, por favor descargar el output.zip, eliminar la carpeta output y reemplzarla con el contenido del .zip. (Poco probable)

## Requisitos

- Python 3.9+ (probado en 3.10)
- Dependencias en `requirements.txt` (incluye torch, transformers, pandas, numpy)
- Opcional: `iterstrat` para split estratificado multilabel

Instalación (Unix/macOS):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Instalación (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Comandos principales

Entrenamiento (ejemplo — Etapa B3, full fine‑tune):

```bash
python train.py --data_path src/data/train.csv --epochs 6 --early_stop 2 --batch_size 16 --lr 1e-5 --att_dim 192 --dropout 0.2 --output_dir outputs/b3
```

Evaluación rápida sobre CSV anotado (usa encoder local si lo deseas):

```bash
python run_eval.py --csv src/data/medical_lit.csv --batch_size 16 --max_len 256 --encoder src/model/local_pubmedbert --model outputs/b3/final_model.pt --thresholds outputs/b3/thresholds.npy --labels outputs/b3/labels.json --save eval_full.csv
```

Evaluación en hold-out (como módulo para evitar problemas de import):

```bash
python -m src.eval.eval_holdout --csv src/data/test.csv --model outputs/b3/final_model.pt --thresholds outputs/b3/thresholds.npy --labels outputs/b3/labels.json --encoder src/model/local_pubmedbert --batch_size 16 --max_len 256
```

Inferencia CLI (ejemplo):

```bash
python run_eval.py --csv src/data/medical_lit.csv --encoder src/model/local_pubmedbert --model outputs/b3/final_model.pt --thresholds outputs/b3/thresholds.npy --labels outputs/b3/labels.json
```

Streamlit UI:

```bash
streamlit run app.py
```

## Notas prácticas y troubleshooting

- Mismatch de `att_dim`: si cargas un `state_dict` entrenado con otra dimensión de atención, el predictor ahora intenta detectar `att_dim` desde el checkpoint y crea el modelo con esa dimensión; si aún hay incompatibilidades, el cargado se hace con `strict=False` y se preservan los pesos compatibles. Si prefieres evitar esto, entrena y evalúa con el mismo `--att_dim`.
- Mezcla de artefactos: usa carpetas por corrida (ej. `outputs/b3/*`) y referencia explícita a `--model`, `--thresholds`, `--labels`.
- OOM en CPU: reduce `--batch_size` (8) y `--max_len` (192–256).
- Imports / ejecución como módulo: para evitar problemas de ruta usa `python -m src.eval.eval_holdout ...`.
- Si usas encoder Hugging Face online, verifica conexión y permisos; para trabajo offline coloca el encoder en `src/model/local_pubmedbert`.

## Buenas prácticas

- Mantén cada corrida en su propia carpeta bajo `outputs/`.
- Guarda también el config (args) usado por cada corrida para reproducibilidad.
- Para cambios en la cabeza (att_dim, dropout) versiona el nombre de la carpeta (p. ej. `b3-att192`).

