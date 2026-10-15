"""Aplicación Streamlit para predecir el abandono de clientes con el modelo entrenado en el notebook."""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
from PIL import Image

from iconos import icono

RUTA_MODELO = Path(__file__).parent / 'modelo_abandono.joblib'
UMBRAL = 0.5
COLOR_SE_QUEDA = '#2a78d6'
COLOR_ABANDONA = '#eb6834'
ALTO_BLOQUE = 365  # misma altura para los tres bloques del formulario

st.set_page_config(page_title='Predicción de abandono', page_icon=Image.open(Path(__file__).parent / 'icono.png'), layout='wide')

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.tarjeta {border-radius: 12px; padding: 1.2rem 1.4rem; color: white; margin-bottom: 1rem;}
.tarjeta h2 {color: white; margin: 0; padding: 0; font-size: 2rem;}
.tarjeta p {margin: 0.2rem 0 0 0; font-size: 1rem; opacity: 0.95;}
.encabezado {display: flex; align-items: center; gap: 0.55rem; font-size: 1.15rem; font-weight: 600; margin-bottom: 0.6rem;}
.titulo-app {display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.3rem;}
.titulo-app h1 {margin: 0; padding: 0;}
.dato {border: 1px solid #e3e8ef; border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.6rem;}
.dato span {display: block; font-size: 0.85rem; color: #5b6472;}
.dato b {font-size: 1.6rem; color: #1f2937;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Diccionarios: lo que ve el usuario (español) y el valor que espera el modelo
# ---------------------------------------------------------------------------
SI_NO = {'No': 'No', 'Sí': 'Yes'}
SERVICIO_INTERNET = {'Fibra óptica': 'Fiber optic', 'DSL': 'DSL', 'Sin internet': 'No'}
TIPO_CONTRATO = {'Mes a mes': 'Month-to-month', 'Un año': 'One year', 'Dos años': 'Two year'}
METODO_PAGO = {
    'Cheque electrónico': 'Electronic check',
    'Cheque por correo': 'Mailed check',
    'Transferencia bancaria (automática)': 'Bank transfer (automatic)',
    'Tarjeta de crédito (automática)': 'Credit card (automatic)',
}
GENERO = {'Mujer': 'Female', 'Hombre': 'Male'}
SERVICIOS_INTERNET = {
    'Seguridad_Online': 'Seguridad en línea',
    'Respaldo_Online': 'Respaldo en línea',
    'Proteccion_Dispositivo': 'Protección del dispositivo',
    'Soporte_Tecnico': 'Soporte técnico',
    'Transmision_TV': 'Televisión por streaming',
    'Transmision_Peliculas': 'Películas por streaming',
}
NOMBRES = {
    'Genero': 'Género',
    'Adulto_Mayor': 'Adulto mayor',
    'Tiene_Pareja': 'Tiene pareja',
    'Tiene_Dependientes': 'Tiene dependientes',
    'Meses_Permanencia': 'Meses de permanencia',
    'Servicio_Telefonico': 'Servicio telefónico',
    'Lineas_Multiples': 'Líneas múltiples',
    'Servicio_Internet': 'Servicio de internet',
    **SERVICIOS_INTERNET,
    'Tipo_Contrato': 'Tipo de contrato',
    'Facturacion_Sin_Papel': 'Facturación sin papel',
    'Metodo_Pago': 'Método de pago',
    'Cargos_Mensuales': 'Cargo mensual',
    'Cargos_Totales': 'Cargos totales',
}
NUMERICAS = ['Adulto_Mayor', 'Meses_Permanencia', 'Cargos_Mensuales', 'Cargos_Totales']


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(ruta):
    """Carga el pipeline y detecta si fue entrenado con otra versión de scikit-learn."""
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        modelo = joblib.load(ruta)
    version_entrenamiento = None
    for aviso in avisos:
        version_entrenamiento = getattr(aviso.message, 'original_sklearn_version', version_entrenamiento)
    return modelo, version_entrenamiento


def variables_del_modelo(modelo):
    """Columnas que usa el preprocesamiento del pipeline, en su orden."""
    pre = modelo.named_steps['preprocesamiento']
    return [col for nombre, _, cols in pre.transformers_ if nombre != 'remainder' for col in cols]


def aportes_por_variable(modelo, cliente):
    """Cuánto sube (+) o baja (-) cada variable la tendencia a abandonar, frente a un cliente promedio.

    Numéricas y ordinales ya están centradas por el escalado; en las nominales se compara
    la categoría elegida con el promedio de las categorías de esa variable.
    """
    pre = modelo.named_steps['preprocesamiento']
    coef = modelo.named_steps['modelo'].coef_[0]
    x = pre.transform(cliente)[0]
    nombres = pre.get_feature_names_out()
    aportes = {}
    for nombre, pipe, cols in pre.transformers_:
        if nombre == 'remainder' or len(cols) == 0:
            continue
        idx = [i for i, n in enumerate(nombres) if n.startswith(nombre + '__')]
        c, v = coef[idx], x[idx]
        if nombre == 'nominales':
            inicio = 0
            for col, categorias in zip(cols, pipe.named_steps['codificar'].categories_):
                fin = inicio + len(categorias)
                aportes[col] = float(c[inicio:fin] @ v[inicio:fin] - c[inicio:fin].mean())
                inicio = fin
        else:
            for col, ci, vi in zip(cols, c, v):
                aportes[col] = float(ci * vi)
    return pd.Series(aportes)


def grafica_aportes(aportes, cantidad=8):
    principales = aportes.reindex(aportes.abs().sort_values(ascending=False).index).head(cantidad)[::-1]
    colores = [COLOR_ABANDONA if a > 0 else COLOR_SE_QUEDA for a in principales]

    plt.figure(figsize=(6, 0.5 * len(principales) + 1))
    plt.barh([NOMBRES.get(v, v) for v in principales.index], principales, color=colores)
    plt.axvline(0, color='gray', linewidth=1)
    plt.xlabel('Menos riesgo  <-->  Más riesgo', fontsize=11)
    plt.yticks(fontsize=11)
    plt.xticks(fontsize=9)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    figura = plt.gcf()
    plt.close()
    return figura


def nivel_de_riesgo(probabilidad):
    if probabilidad >= 0.7:
        return 'Muy alto'
    if probabilidad >= UMBRAL:
        return 'Alto'
    if probabilidad >= 0.3:
        return 'Medio'
    return 'Bajo'


# ---------------------------------------------------------------------------
# Controles de entrada
# ---------------------------------------------------------------------------
def opcion(etiqueta, opciones, clave, por_defecto=0, horizontal=True):
    """Radio con textos en español que devuelve el valor que entiende el modelo."""
    elegido = st.radio(etiqueta, list(opciones), index=por_defecto, horizontal=horizontal, key=clave)
    return opciones[elegido]


def lista(etiqueta, opciones, clave, por_defecto=0):
    elegido = st.selectbox(etiqueta, list(opciones), index=por_defecto, key=clave)
    return opciones[elegido]


def formulario_cliente(usa):
    """Dibuja solo los campos que usa el modelo y devuelve el cliente como DataFrame de una fila."""
    datos = {}
    col_perfil, col_contrato, col_pagos = st.columns(3)

    with col_perfil:
        with st.container(border=True, height=ALTO_BLOQUE):
            encabezado('usuario', 'Perfil del cliente')
            if 'Genero' in usa:
                datos['Genero'] = opcion('Género', GENERO, 'genero')
            if 'Adulto_Mayor' in usa:
                datos['Adulto_Mayor'] = 1 if opcion('Adulto mayor (65 años o más)', SI_NO, 'mayor') == 'Yes' else 0
            if 'Tiene_Pareja' in usa:
                datos['Tiene_Pareja'] = opcion('Tiene pareja', SI_NO, 'pareja')
            if 'Tiene_Dependientes' in usa:
                datos['Tiene_Dependientes'] = opcion('Tiene dependientes', SI_NO, 'dependientes')

    with col_contrato:
        with st.container(border=True, height=ALTO_BLOQUE):
            encabezado('contrato', 'Contrato')
            datos['Meses_Permanencia'] = st.slider('Meses de permanencia', 0, 72, 12, key='meses')
            datos['Tipo_Contrato'] = lista('Tipo de contrato', TIPO_CONTRATO, 'contrato')
            datos['Facturacion_Sin_Papel'] = opcion('Facturación sin papel', SI_NO, 'papel', por_defecto=1)

    with col_pagos:
        with st.container(border=True, height=ALTO_BLOQUE):
            encabezado('tarjeta', 'Pagos')
            datos['Metodo_Pago'] = lista('Método de pago', METODO_PAGO, 'pago')
            datos['Cargos_Mensuales'] = st.number_input(
                'Cargo mensual (USD)', min_value=0.0, max_value=200.0, value=70.0, step=5.0, key='mensual')
            automatico = st.toggle('Calcular cargos totales (meses x cargo mensual)', value=True, key='auto')
            if automatico:
                meses = datos['Meses_Permanencia']
                datos['Cargos_Totales'] = round(meses * datos['Cargos_Mensuales'], 2) if meses > 0 else None
                texto = f"{datos['Cargos_Totales']:,.2f} USD" if meses > 0 else 'vacío (cliente nuevo)'
                st.caption(f'Cargos totales: **{texto}**')
            else:
                datos['Cargos_Totales'] = st.number_input(
                    'Cargos totales (USD)', min_value=0.0, max_value=10000.0, value=840.0, step=50.0, key='totales')

    with st.container(border=True):
        encabezado('wifi', 'Servicios contratados')
        if 'Servicio_Telefonico' in usa:
            col_tel, col_lineas = st.columns(2)
            with col_tel:
                datos['Servicio_Telefonico'] = opcion('Servicio telefónico', SI_NO, 'telefono', por_defecto=1)
            with col_lineas:
                if datos['Servicio_Telefonico'] == 'Yes':
                    datos['Lineas_Multiples'] = opcion('Líneas múltiples', SI_NO, 'lineas')
                else:
                    datos['Lineas_Multiples'] = 'No phone service'
        elif 'Lineas_Multiples' in usa:
            datos['Lineas_Multiples'] = opcion('Líneas múltiples', SI_NO, 'lineas')

        datos['Servicio_Internet'] = opcion('Servicio de internet', SERVICIO_INTERNET, 'internet')
        if datos['Servicio_Internet'] == 'No':
            for col in SERVICIOS_INTERNET:
                datos[col] = 'No internet service'
            st.caption('Sin internet: los servicios adicionales no aplican.')
        else:
            columnas = st.columns(3)
            for i, (col, etiqueta) in enumerate(SERVICIOS_INTERNET.items()):
                with columnas[i % 3]:
                    datos[col] = opcion(etiqueta, SI_NO, col)

    return pd.DataFrame([{col: datos.get(col) for col in usa}])


def encabezado(nombre_icono, texto):
    st.markdown(f'<div class="encabezado">{icono(nombre_icono)}<span>{texto}</span></div>',
                unsafe_allow_html=True)


def dato(etiqueta, valor):
    st.markdown(f'<div class="dato"><span>{etiqueta}</span><b>{valor}</b></div>', unsafe_allow_html=True)


@st.dialog('Resultado de la predicción', width='large')
def ventana_resultado(modelo, cliente):
    probabilidad = float(modelo.predict_proba(cliente)[0, 1])
    abandona = probabilidad >= UMBRAL
    color = COLOR_ABANDONA if abandona else COLOR_SE_QUEDA
    titulo = 'Abandona' if abandona else 'Se queda'
    mensaje = ('El cliente tiene alta probabilidad de cancelar el servicio.' if abandona
               else 'El cliente probablemente mantendrá el servicio.')

    col_resultado, col_grafica = st.columns([2, 3], gap='large')
    with col_resultado:
        st.markdown(f"""
        <div class="tarjeta" style="background:{color};">
            <p>Predicción</p>
            <h2>{titulo}</h2>
            <p>{mensaje}</p>
        </div>""", unsafe_allow_html=True)
        dato('Probabilidad de abandono', f'{probabilidad:.0%}')
        st.progress(probabilidad)
        dato('Nivel de riesgo', nivel_de_riesgo(probabilidad))
        st.caption(f'Se marca como "abandona" cuando la probabilidad es {UMBRAL:.0%} o más.')

    with col_grafica:
        encabezado('grafica', '¿Qué pesa en esta predicción?')
        st.pyplot(grafica_aportes(aportes_por_variable(modelo, cliente)))
        st.caption('Naranja: acerca al abandono. Azul: ayuda a retener. Se compara con un cliente promedio.')

    with st.expander('Ver los datos que recibe el modelo'):
        st.dataframe(cliente.T.astype(str).rename(columns={0: 'Valor'}), width='stretch')


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
def pestana_individual(modelo, usa):
    cliente = formulario_cliente(usa)
    _, centro, _ = st.columns([1, 2, 1])
    if centro.button('Predecir abandono', type='primary', width='stretch', key='predecir'):
        ventana_resultado(modelo, cliente)


def ejemplo_csv(usa):
    ejemplo = {
        'Genero': 'Female', 'Adulto_Mayor': 0, 'Tiene_Pareja': 'No', 'Tiene_Dependientes': 'No',
        'Meses_Permanencia': 2, 'Servicio_Telefonico': 'Yes', 'Lineas_Multiples': 'No',
        'Servicio_Internet': 'Fiber optic', 'Seguridad_Online': 'No', 'Respaldo_Online': 'No',
        'Proteccion_Dispositivo': 'No', 'Soporte_Tecnico': 'No', 'Transmision_TV': 'Yes',
        'Transmision_Peliculas': 'Yes', 'Tipo_Contrato': 'Month-to-month', 'Facturacion_Sin_Papel': 'Yes',
        'Metodo_Pago': 'Electronic check', 'Cargos_Mensuales': 95.5, 'Cargos_Totales': 191.0,
    }
    tabla = pd.DataFrame([{'ID_Cliente': 'CLIENTE-001', **{col: ejemplo[col] for col in usa}}])
    return tabla.to_csv(index=False).encode('utf-8-sig')


def pestana_lote(modelo, usa):
    st.markdown('Sube un archivo CSV con **una fila por cliente** y las columnas que usa el modelo. '
                'Las columnas adicionales (por ejemplo un ID) se conservan en el resultado.')
    st.download_button('Descargar plantilla CSV', ejemplo_csv(usa), 'plantilla_clientes.csv', 'text/csv')

    archivo = st.file_uploader('Archivo CSV de clientes', type='csv')
    if archivo is None:
        return

    clientes = pd.read_csv(archivo)
    faltantes = [col for col in usa if col not in clientes.columns]
    if faltantes:
        st.error('Faltan estas columnas en el archivo: ' + ', '.join(faltantes))
        return

    for col in NUMERICAS:
        if col in clientes.columns:
            clientes[col] = pd.to_numeric(clientes[col], errors='coerce')

    # Las variables con orden (Tipo_Contrato) solo aceptan sus valores conocidos
    pre = modelo.named_steps['preprocesamiento']
    for nombre, pipe, cols in pre.transformers_:
        if nombre != 'ordinales':
            continue
        for col, validos in zip(cols, pipe.named_steps['codificar'].categories_):
            invalidas = clientes.index[clientes[col].notna() & ~clientes[col].isin(validos)]
            if len(invalidas):
                filas = ', '.join(str(i + 2) for i in invalidas[:10])
                st.error(f'La columna **{col}** tiene valores no válidos en las filas {filas} del archivo. '
                         f'Valores permitidos: {", ".join(validos)}.')
                return

    try:
        probabilidades = modelo.predict_proba(clientes[usa])[:, 1]
    except ValueError as error:
        st.error(f'El archivo tiene valores que el modelo no reconoce: {error}')
        return

    resultado = clientes.copy()
    resultado['Probabilidad_Abandono'] = probabilidades.round(3)
    resultado['Prediccion'] = ['Abandona' if p >= UMBRAL else 'Se queda' for p in probabilidades]
    resultado = resultado.sort_values('Probabilidad_Abandono', ascending=False)

    en_riesgo = int((probabilidades >= UMBRAL).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric('Clientes evaluados', f'{len(resultado):,}')
    c2.metric('Marcados como "abandona"', f'{en_riesgo:,}')
    c3.metric('Porcentaje en riesgo', f'{en_riesgo / len(resultado):.0%}')

    st.dataframe(
        resultado, width='stretch', hide_index=True,
        column_config={'Probabilidad_Abandono': st.column_config.ProgressColumn(
            'Probabilidad de abandono', min_value=0, max_value=1, format='%.2f')},
    )
    st.download_button('Descargar resultados (CSV)', resultado.to_csv(index=False).encode('utf-8-sig'),
                       'predicciones_abandono.csv', 'text/csv', type='primary')


def barra_lateral(usa, version_entrenamiento):
    with st.sidebar:
        st.markdown('### Sobre el modelo')
        st.markdown(f"""
- **Algoritmo:** regresión logística
- **Variables de entrada:** {len(usa)}
- **Detecta** el **80 %** de los clientes que abandonan
- De los marcados como "abandona", el **51 %** realmente abandona
""")
        st.divider()
        st.markdown('### Versión')
        st.markdown(f'scikit-learn instalado: **{sklearn.__version__}**')
        if version_entrenamiento:
            st.warning(f'El modelo se entrenó con scikit-learn **{version_entrenamiento}**. '
                       f'Instala esa versión para evitar resultados incorrectos:\n\n'
                       f'pip install scikit-learn=={version_entrenamiento}')
        else:
            st.success('El modelo es compatible con la versión instalada.')
        st.divider()
        st.caption('Datos: Telco Customer Churn (Kaggle). Modelo entrenado en el notebook abandon_clientes.ipynb.')
        st.caption('Autores: ' + ' y '.join(AUTORES) + '.')
        if st.button('Volver a la bienvenida', width='stretch'):
            st.session_state.ingreso = False
            st.rerun()


# ---------------------------------------------------------------------------
# Bienvenida
# ---------------------------------------------------------------------------
AUTORES = ['Omar Alberto Torres', 'Diego Alejandro Roa Lara']
PROGRAMA = 'Maestría en Análisis de Datos y Sistemas Inteligentes'
FECHA = 'Noviembre de 2026'


def pantalla_bienvenida():
    st.markdown("""
    <style>
    .portada {text-align: center; padding: 2.5rem 1rem 1rem 1rem;}
    .portada .icono {margin-bottom: 0.3rem;}
    .portada h1 {font-size: 2.6rem; margin: 0.4rem 0;}
    .portada .sub {font-size: 1.15rem; color: #5b6472; max-width: 720px; margin: 0 auto;}
    .funcion {border: 1px solid #e3e8ef; border-radius: 12px; padding: 1.1rem; height: 100%; text-align: center;}
    .funcion .icono {margin-bottom: 0.2rem;}
    .funcion b {display: block; margin: 0.3rem 0; font-size: 1.05rem;}
    .funcion span {color: #5b6472; font-size: 0.92rem;}
    .creditos {background: #f3f6fa; border-radius: 12px; padding: 1.4rem; text-align: center; margin-top: 1.6rem;}
    .creditos .titulo {font-size: 0.85rem; letter-spacing: 0.08em; color: #2a78d6; font-weight: 600;}
    .creditos .autores {font-size: 1.3rem; font-weight: 600; margin: 0.5rem 0; color: #1f2937;}
    .creditos .programa {color: #5b6472;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="portada">
        <div class="icono">{icono("tendencia", 64)}</div>
        <h1>Predicción de abandono de clientes</h1>
        <p class="sub">Aplicación que estima la probabilidad de que un cliente de telecomunicaciones
        cancele el servicio, usando un modelo de regresión logística entrenado con datos reales.</p>
    </div>
    """, unsafe_allow_html=True)

    _, centro, _ = st.columns([1, 4, 1])
    with centro:
        funciones = [
            ('usuario', 'Un cliente', 'Ingresa sus datos y obtén la predicción al instante.'),
            ('usuarios', 'Varios clientes', 'Sube un archivo CSV y descarga las predicciones.'),
            ('lupa', 'Explicación', 'Mira qué variables pesaron más en cada resultado.'),
        ]
        for columna, (icono_nombre, titulo, texto) in zip(st.columns(3), funciones):
            columna.markdown(f'<div class="funcion"><div class="icono">{icono(icono_nombre, 32)}</div>'
                             f'<b>{titulo}</b><span>{texto}</span></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="creditos">
            <div>{icono("birrete", 30)}</div>
            <div class="titulo">DESARROLLADO POR</div>
            <div class="autores">{'<br>'.join(AUTORES)}</div>
            <div class="programa">Estudiantes de la {PROGRAMA}<br>{FECHA}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write('')
        _, boton, _ = st.columns([1, 2, 1])
        if boton.button('Continuar', type='primary', width='stretch', key='continuar'):
            st.session_state.ingreso = True
            st.rerun()


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
if not st.session_state.get('ingreso', False):
    pantalla_bienvenida()
    st.stop()

st.markdown(f'<div class="titulo-app">{icono("tendencia", 44)}<h1>Predicción de abandono de clientes</h1></div>',
            unsafe_allow_html=True)
st.markdown('Ingresa los datos de un cliente y el modelo estimará la **probabilidad de que cancele el servicio**.')

if not RUTA_MODELO.exists():
    st.error(f'No se encontró el archivo del modelo en **{RUTA_MODELO}**.\n\n'
             'Descárgalo desde la sección 5.1 del notebook y guárdalo en la misma carpeta que app.py.')
    st.stop()

modelo, version_entrenamiento = cargar_modelo(str(RUTA_MODELO))
usa = variables_del_modelo(modelo)
barra_lateral(usa, version_entrenamiento)

tab_individual, tab_lote = st.tabs(['Un cliente', 'Varios clientes (CSV)'])
with tab_individual:
    pestana_individual(modelo, usa)
with tab_lote:
    pestana_lote(modelo, usa)
