import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd

# --- Cargar el modelo Keras ---
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model("modelo_autoencoder1.keras")
    return modelo

# --- Preprocesamiento para imágenes en escala de grises ---
def preparar_imagen_autoencoder(imagen):
    imagen = imagen.convert("L")  # Convertir a escala de grises (1 canal)
    imagen = imagen.resize((224, 224))
    matriz = np.array(imagen).astype(np.float32) / 255.0
    matriz = np.expand_dims(matriz, axis=0)       # Añadir dimensión de batch
    matriz = np.expand_dims(matriz, axis=-1)      # Añadir canal (1)
    return matriz

# --- Etiquetas del modelo ---
etiquetas = [
    "ARP_Spoofing", "Backdoor_Malware", "Browser_Hijacking", "Brute_Force",
    "Command_Injection", "DDOS", "DNS_Spoofing", "DOS", "Mirai",
    "MITM_ARP_Spoofing", "Normal", "Scan", "SqlInjection", "Theft"
]

# --- Título de la aplicación ---
st.title("🧠 Clasificador de Ciberataques con Keras")
st.write("Sube una imagen para predecir su categoría de ciberataque.")

# --- Cargar imagen del usuario ---
archivo_imagen = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo_imagen:
    imagen = Image.open(archivo_imagen)
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    imagen_preparada = preparar_imagen_autoencoder(imagen)

    # --- Cargar modelo e inferencia ---
    modelo = cargar_modelo()
    salida_predicha = modelo.predict(imagen_preparada)

    # Si el modelo retorna una lista de outputs, usar la salida que contiene las predicciones de clase
    if isinstance(salida_predicha, list):
        salida_predicha = salida_predicha[1]  # <-- Esta es la que tiene softmax normalmente

    clase = int(np.argmax(salida_predicha))
    confianza = float(np.max(salida_predicha))

    # Intentamos extraer la etiqueta con un try/except
    try:
        ataque_predicho = etiquetas[clase]
    except IndexError:
        st.error(f"Error: clase predicha ({clase}) fuera de rango al asignar etiqueta.")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al asignar etiqueta: {e}")
        st.stop()

    st.success(f"🧠 Predicción: **{ataque_predicho}**")
    st.info(f"📊 Confianza del modelo: **{confianza * 100:.2f}%**")

    # --- Visualización opcional ---
    if st.checkbox("Mostrar probabilidades por clase"):
        st.bar_chart(salida_predicha[0])

# --- Cargar metadatos de especies ---
@st.cache_data
def cargar_metadatos():
    return pd.read_excel("ataques.xlsx")

# --- Mostrar información complementaria ---
df_metadatos = cargar_metadatos()

# Limpiar posibles espacios o comillas en los headers
df_metadatos.columns = (
    df_metadatos.columns
      .str.strip()
      .str.strip("'\"")
)

# Volvemos a proteger el acceso a etiquetas[clase] por si clase no existe
try:
    ataque_predicho  # Verifica que existe (si no, KeyError/NameError)
except Exception:
    st.info("Aún no se ha hecho una predicción válida.")
else:
    try:
        # Intentamos extraer etiqueta de nuevo (en caso de entrar aquí sin if archivo_imagen)
        ataque_predicho = etiquetas[clase]
    except Exception:
        st.error("No se pudo recuperar la etiqueta predicha para buscar metadatos.")
        st.stop()

    # Intentamos filtrar el DataFrame sobre la columna 'Nombre'
    try:
        filtro = df_metadatos[df_metadatos['Nombre'] == ataque_predicho]
    except KeyError:
        st.error("El DataFrame no contiene la columna 'Nombre'. Revisa los encabezados del Excel.")
        st.stop()

    if not filtro.empty:
        st.subheader("📚 Información adicional del Ataque:")
        st.write(f"**Nombre completo:** {filtro.iloc[0]['Nombre']}")
        st.markdown(f"**Descripción:** {filtro.iloc[0]['Descripcion']}")
        st.markdown(f"**Presente en la Capa:** {filtro.iloc[0]['Capa']}")
    else:
        st.warning(f"No se encontró información complementaria para **{ataque_predicho}**.")
