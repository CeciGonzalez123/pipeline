#!/bin/bash
# clean.sh - Procesa un archivo de texto con clean.py

# Verificar que se proporcionó un archivo de entrada
if [ -z "$1" ]; then
    echo "Error: Debes especificar la ruta del archivo a limpiar"
    echo "Uso: $0 <archivo_entrada> [archivo_salida]"
    exit 1
fi

# Establecer archivo de salida (corpus por defecto)
ARCHIVO_SALIDA=${2:-"corpus"}

# Ruta al script de limpieza (ajusta según tu estructura)
SCRIPT_LIMPIEZA="methods/clean/clean.py"

# Ejecutar el proceso de limpieza
echo "Procesando $1 -> $ARCHIVO_SALIDA"
python3 "$SCRIPT_LIMPIEZA" "$1" "$ARCHIVO_SALIDA"