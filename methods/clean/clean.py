#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de Limpieza para Corpus en Guaraní

Este script implementa un pipeline completo para la limpieza y normalización de corpus de texto en guaraní,
siguiendo las especificaciones del proyecto UC Autumn of Code 2025. El proceso incluye:

1. Filtrado básico de documentos según criterios de longitud y contenido
2. Normalización de texto (codificación, caracteres especiales, formato)
3. Validación de relevancia lingüística
4. Almacenamiento del corpus procesado

Características principales:
- Procesamiento eficiente de grandes volúmenes de texto
- Manejo adecuado de caracteres especiales del guaraní
- Compatibilidad con Python 3.9+
- Total adherencia a PEP 8 y documentación completa

"""

import re
import os
import sys
from pathlib import Path
from typing import Tuple, List, Pattern


class Limpiador:
    """Clase principal para el procesamiento y limpieza de corpus en guaraní.

    Atributos:
        oraciones_min (int): Número mínimo de oraciones por documento
        palabras_min (int): Número mínimo de palabras por línea
        patrones_irrelevantes (List[str]): Lista de patrones a filtrar
        patrones_compilados (List[Pattern]): Patrones compilados para búsqueda
        patron_division_oraciones (Pattern): Patrón para dividir oraciones
    """

    PATRONES_IRRELEVANTES = [
        r"lorem ipsum",
        r"javascript",
        r"cookie policy",
        r"política de cookies",
        r"terms of service",
        r"términos de servicio",
        r"privacy policy",
        r"política de privacidad"
    ]

    def __init__(self, oraciones_min: int = 3, palabras_min: int = 2) -> None:
        """Inicializa el limpiador con parámetros configurables.

        Args:
            oraciones_min: Mínimo de oraciones requeridas por documento (default: 3)
            palabras_min: Mínimo de palabras requeridas por línea (default: 2)
        """
        self.oraciones_min = oraciones_min
        self.palabras_min = palabras_min
        self.patrones_irrelevantes = self.PATRONES_IRRELEVANTES
        
        self.patrones_compilados = [
            re.compile(patron, re.IGNORECASE) 
            for patron in self.patrones_irrelevantes
        ]
        
        self.patron_division_oraciones = re.compile(
            r'([.!?…]|Héẽ|Maitei)\s+'
        )

    def es_relevante(self, texto: str) -> bool:
        """Determina si un texto cumple los criterios básicos de calidad.

        Args:
            texto: Cadena de texto a evaluar

        Returns:
            bool: True si el texto es relevante, False si debe filtrarse

        Ejemplo:
            >>> limpiador = LimpiadorGuarani()
            >>> limpiador.es_relevante("Hola. Esto es guaraní. Maitei.")
            True
        """
        for patron in self.patrones_compilados:
            if patron.search(texto):
                return False
                
        partes = self.patron_division_oraciones.split(texto)
        oraciones = []
        buffer = ""
        
        for i, parte in enumerate(partes):
            if i % 2 == 0:
                buffer = parte
            else:
                buffer += parte[0]
                oraciones.append(buffer.strip())
                buffer = ""
        
        if buffer:
            oraciones.append(buffer.strip())
            
        oraciones = [o for o in oraciones if o]
        
        if len(oraciones) < self.oraciones_min:
            return False
            
        lineas = texto.split('\n')
        for linea in lineas:
            palabras = linea.split()
            if len(palabras) > 0 and len(palabras) < self.palabras_min:
                return False
                
        return True

    def normalizar_texto(self, texto: str) -> str:
        """Normaliza el texto aplicando transformaciones estándar.

        Args:
            texto: Texto crudo a normalizar

        Returns:
            str: Texto normalizado

        Procesos aplicados:
            - Eliminación de etiquetas HTML
            - Normalización de espacios
            - Corrección de caracteres especiales
            - Normalización de caracteres guaraníes
        """
        texto = re.sub(r'<[^>]+>', ' ', texto)
        texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        reemplazos = [
            (r'[g̃G̃]', 'ĝ'),
            (r'[ãÃ]', 'ã'),
            (r'[ẽẼ]', 'ẽ'),
            (r'[ĩĨ]', 'ĩ'),
            (r'[õÕ]', 'õ'),
            (r'[ũŨ]', 'ũ'),
            (r'[ýÝ]', 'ý'),
            (r'[´`]', "'"),
            (r'[“”]', '"'),
        ]
        
        for patron, reemplazo in reemplazos:
            texto = re.sub(patron, reemplazo, texto)
            
        return texto

    def limpiar_documento(self, texto: str) -> Tuple[bool, str]:
        """Ejecuta el pipeline completo de limpieza sobre un documento.

        Args:
            texto: Contenido crudo del documento

        Returns:
            Tuple[bool, str]: (se_conserva, texto_limpio)
        """
        normalizado = self.normalizar_texto(texto)
        return self.es_relevante(normalizado), normalizado

    def procesar_archivo(
        self, 
        ruta_entrada: str, 
        directorio_salida: str
    ) -> None:
        """Procesa un archivo completo y guarda el resultado.

        Args:
            ruta_entrada: Ruta al archivo de entrada
            directorio_salida: Directorio para archivos procesados

        Raises:
            IOError: Si hay problemas al leer/escribir archivos
        """
        try:
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                contenido = f.read()
        except UnicodeDecodeError:
            try:
                with open(ruta_entrada, 'r', encoding='latin-1') as f:
                    contenido = f.read()
            except Exception as e:
                print(f"Error al leer archivo {ruta_entrada}: {str(e)}")
                return
        
        se_conserva, contenido_limpio = self.limpiar_documento(contenido)
        
        if se_conserva:
            os.makedirs(directorio_salida, exist_ok=True)
            nombre_archivo = Path(ruta_entrada).stem
            ruta_salida = os.path.join(
                directorio_salida, 
                f"{nombre_archivo}_limpio.txt"
            )
            
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                f.write(contenido_limpio)
            print(f"Procesado exitoso: {ruta_salida}")
        else:
            print(f"Documento no cumple los criterios: {ruta_entrada}")


def main() -> None:
    """Función principal para ejecución desde línea de comandos.
    
    Uso esperado:
        python3 clean.py archivo_entrada.txt directorio_salida/
    """
    if len(sys.argv) != 3:
        print("Uso: python3 clean.py archivo_entrada.txt directorio_salida/")
        sys.exit(1)
        
    limpiador = Limpiador()
    limpiador.procesar_archivo(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
