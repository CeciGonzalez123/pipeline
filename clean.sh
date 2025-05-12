#!/bin/bash
# clean.sh - Procesa archivos .txt de la carpeta indicada con methods/clean.py

# Verificar que se proporcionó una carpeta de entrada
if [ -z "$1" ]; then
    echo "Error: Debes especificar la ruta de la carpeta a procesar"
    echo "Uso: $0 <directorio_entrada> [dirctorio_salida]"
    exit 1
fi

# Establecer archivo de salida (corpus por defecto)
ARCHIVO_SALIDA=${2:-"corpus"}

# Ruta al script de limpieza (ajustar según la estructura)
SCRIPT_LIMPIEZA="methods/clean/clean.py"

# Ejecutar el proceso de limpieza
echo "Procesando $1 -> $ARCHIVO_SALIDA"
python3 "$SCRIPT_LIMPIEZA" -i "$1" -o "$ARCHIVO_SALIDA" $3
