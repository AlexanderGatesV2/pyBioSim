

# pyBioSim

Una simulación evolutiva de criaturas con redes neuronales, compatible con biosim4.

## Descripción

pyBioSim es un entorno de simulación donde criaturas virtuales con redes neuronales evolucionan a lo largo de las generaciones. Las criaturas navegan por un mundo 2D, respondiendo a desafíos ambientales a través de sus redes neuronales, las cuales están codificadas por genomas que evolucionan mediante selección natural.

Este proyecto es una implementación en Python de la simulación evolutiva en C++ [biosim4](https://github.com/davidrmiller/biosim4), con mejoras para la visualización y la interactividad.

## Características

-  Criaturas controladas por redes neuronales
-  Algoritmo genético con cruce y mutación
-  Diversos desafíos ambientales
-  Visualización interactiva con pygame
-  Parámetros de simulación personalizables
-  Herramientas de registro y análisis de datos
-  Compatibilidad con archivos de configuración de biosim4 en C++

## Compatibilidad con C++

Este proyecto se está actualizando para garantizar la compatibilidad con la implementación original de biosim4 en C++. El objetivo es recrear fielmente el comportamiento de la simulación de C++ mientras se mantienen las ventajas de la arquitectura de Python.

### Estado de compatibilidad

-  [x] Implementación de genes y genomas
-  [x] Implementación de redes neuronales
-  [x] Implementación de movimiento y sensores
-  [x] Supervivencia y reproducción
-  [x] Cuadrícula y entorno
-  [x] Integración y verificación
-  [x] Documentación y finalización

## Instalación

### Instalación rápida

#### En Linux/macOS:

```bash
# Make the script executable if needed
chmod +x install.sh

# Run the installation script
./install.sh
```

#### En Windows:

```
# Run the installation script
install.bat
```

Los scripts de instalación se encargarán de:

1. Crear opcionalmente un entorno virtual
2. Instalar todas las dependencias requeridas
3. Instalar el paquete en modo de desarrollo

### Instalación manual

#### Usando pip

```bash
# Install from the current directory
pip install .

# Or install in development mode
pip install -e .
```

#### Paso a paso

1. Clona el repositorio
2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate     # On Windows
   ```
3. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```
4. Instala el paquete:
   ```bash
   pip install -e .
   ```

## Uso

### Ejecutar la simulación

```bash
# Run with default parameters
python main.py

# Run with a custom configuration file
python main.py --config custom_config.json
```

### Controles

-  **ESPACIO**: Pausar/Reanudar la simulación
-  **S**: Alternar resaltado del área de desafío
-  **D**: Alternar líneas de dirección
-  **K**: Alternar visualización del contador de muertes
-  **0-3**: Cambiar tipo de barrera
-  **+/-**: Ajustar velocidad de la simulación
-  **G**: Forzar nueva generación (cuando está en pausa)
-  **R**: Reiniciar simulación (cuando está en pausa)
-  **F1**: Alternar visualización de ayuda

### Desafíos

La simulación incluye diversos desafíos ambientales que determinan qué criaturas sobreviven y se reproducen:

-  **Desafío Circular (0)**: Las criaturas deben permanecer dentro de un área circular en el cuadrante superior izquierdo.
-  **Mitad Derecha (1)**: Las criaturas deben permanecer en la mitad derecha de la arena.
-  **Cuarto Derecho (2)**: Las criaturas deben permanecer en el cuarto más a la derecha de la arena.
-  **Cadena (3)**: Las criaturas deben tener 2-3 vecinos dentro de un radio de 1.5 celdas.
-  **Centro Ponderado (4)**: Las criaturas deben permanecer cerca del centro, con puntuaciones más altas para aquellas más cercanas al centro.
-  **Centro No Ponderado (19)**: Las criaturas deben permanecer cerca del centro, con puntuaciones iguales para todos los sobrevivientes.
-  **Esquina (5)**: Las criaturas deben permanecer cerca de cualquiera de las esquinas de la arena.
-  **Esquina Ponderada (6)**: Las criaturas deben permanecer cerca de cualquier esquina, con puntuaciones más altas para aquellas más cercanas a las esquinas.
-  **Distancia de Migración (7)**: Las criaturas son puntuadas según la distancia recorrida desde su posición de nacimiento.
-  **Centro Disperso (8)**: Las criaturas deben permanecer cerca del centro con un conteo específico de vecinos.
-  **Octavo Izquierdo (9)**: Las criaturas deben permanecer en la octava parte más a la izquierda de la arena.
-  **Muros Radiactivos (10)**: Las criaturas deben evitar los muros que se vuelven radiactivos.
-  **Contra Cualquier Muro (11)**: Las criaturas deben tocar cualquiera de los muros de la arena.
-  **Tocar Cualquier Muro (12)**: Las criaturas deben haber tocado un muro durante su vida.
-  **Octavos Este y Oeste (13)**: Las criaturas deben permanecer en el octavo más a la izquierda o más a la derecha.
-  **Cerca de la Barrera (14)**: Las criaturas deben permanecer cerca de las barreras.
-  **Parejas (15)**: Las criaturas deben formar parejas exclusivas con una configuración específica de vecinos.
-  **Secuencia de Ubicación (16)**: Las criaturas son puntuadas según el número de ubicaciones visitadas.
-  **Altruismo (17)**: Las criaturas en la zona segura del noroeste obtienen puntuaciones más altas.
-  **Altruismo y Sacrificio (18)**: Las criaturas en la zona de sacrificio del noreste son seleccionadas para la reproducción basada en parentesco.

**Corrección reciente**: La constante del desafío Centro No Ponderado se cambió de 4 a 19 para corregir un problema en el que tanto el desafío Centro Ponderado como el Centro No Ponderado utilizaban el mismo valor de constante (4). Esto garantiza que los dos tipos de desafío diferentes puedan distinguirse correctamente en el código.

### Contador de muertes

La simulación incluye una función de contador de muertes que rastrea cuántas criaturas son eliminadas durante cada generación. Esta función funciona cuando la neurona de muerte está habilitada en tu configuración.

-  Presiona **K** para activar/desactivar la visualización del contador de muertes
-  Cuando está activado, el conteo de muertes aparece en texto rojo en la parte superior derecha de la pantalla
-  El conteo de muertes también se muestra en la ventana de ayuda (presiona F1)
-  El contador se reinicia al final de cada generación
-  Esta función ayuda a rastrear el comportamiento depredador en la simulación

## Configuración

La simulación puede personalizarse a través de un archivo de configuración JSON. Consulta `config.json` para ver los parámetros disponibles.

### Archivos de configuración de C++

pyBioSim puede leer archivos de configuración .ini de biosim4 en C++ directamente:

```bash
# Run with a C++ biosim4 configuration file
python main.py --config biosim4.ini
```

Esto permite una comparación directa entre las implementaciones de C++ y Python utilizando la misma configuración.

## Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.
