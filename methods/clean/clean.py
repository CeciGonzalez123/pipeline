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
from datasketch import MinHashLSH, MinHash
import json
import re
import os
import sys
from pathlib import Path
from typing import Tuple, List, Pattern
import argparse
from pathlib import Path
import fasttext
import numpy as np
import nltk
from huggingface_hub import hf_hub_download
import hashlib

# Configuración esencial para NLTK
nltk.download('punkt', quiet=True)

class ValidatorGuarani:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.ft_model = fasttext.load_model('lid.176.bin')
        self.glotlid_model = self._load_glotlid()
        
    def _load_glotlid(self):
        model_path = hf_hub_download(
            repo_id="cis-lmu/glotlid",
            filename="model.bin"
        )
        return fasttext.load_model(model_path)

    def _is_guarani(self, text: str) -> bool:
        """Determina si un texto es guaraní con ambos modelos"""
        ft_label, _ = self.ft_model.predict(text, k=1)
        glot_label, _ = self.glotlid_model.predict(text, k=1)
        
        ft_gn = '__label__gn' in ft_label[0]
        glot_gn = any(x in glot_label[0] for x in [
            '__label__grn_Latn', '__label__gug_Latn', '__label__gn'
        ])
        
        return ft_gn and glot_gn

    def validate_content(self, content: str) -> Tuple[bool, float]:
        """Valida contenido de texto en lugar de archivo"""
        try:
            content = content.replace('\n', ' ')
            sentences = nltk.sent_tokenize(content, language='spanish')
            
            if not sentences:
                return (False, 0.0, content)
            
            valid_count = 0
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) >= 5 and self._is_guarani(clean_sentence):
                    valid_count += 1
                    
            percentage = valid_count / len(sentences)
            return (percentage >= self.threshold, percentage, content)
            
        except Exception as e:
            print(f"  ✗ Error validación: {str(e)}")
            return (False, 0.0, content)
        

class ValidadorCalidad:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold  # Umbral global de calidad (0-1)
        
        # Configuración de parámetros ajustables
        self.config = {
            'max_upper_ratio': 0.2,
            'min_words': 3,
            'max_words': 50,
            'repetition_window': 5,
            'max_repetitions': 2,
            'allowed_chars': r'[a-zA-ZãẽĩõũáéíóúýñçÁÉÍÓÚÝÑÇ\'\- \t\n.,!?;:¿¡%&$#@()]'
        }
    
    def validate_document(self, text: str) -> Tuple[bool, float]:
        """Evalúa el texto y devuelve si cumple con el umbral de calidad"""
        scores = []
        
        # 1. Capitalización consistente (20% peso)
        cap_score = self._check_capitalization(text)
        scores.append(cap_score * 0.2)
        
        # 2. Longitud de oraciones (30% peso)
        len_score = self._check_sentence_lengths(text)
        scores.append(len_score * 0.3)
        
        # 3. Repeticiones (25% peso)
        rep_score = self._check_repetitions(text)
        scores.append(rep_score * 0.25)
        
        # 4. Caracteres válidos (25% peso)
        char_score = self._check_suspicious_chars(text)
        scores.append(char_score * 0.25)
        
        total_score = sum(scores)
        return (total_score >= self.threshold, total_score)

    def _check_capitalization(self, text: str) -> float:
        letras = [c for c in text if c.isalpha()]
        if not letras:
            return 1.0  # Texto sin letras no se penaliza
            
        mayusculas = sum(1 for c in letras if c.isupper())
        oraciones = re.split(r'[.!?…]', text)
        validas = sum(1 for o in oraciones if o.strip() and o.strip()[0].isupper())
        
        ratio_invalidas = max(0, (mayusculas - validas) / len(letras))
        return 1.0 - min(ratio_invalidas / self.config['max_upper_ratio'], 1.0)

    def _check_sentence_lengths(self, text: str) -> float:
        oraciones = re.split(r'[.!?…]', text)
        if not oraciones:
            return 0.0
            
        validas = 0
        for o in oraciones:
            palabras = o.strip().split()
            if self.config['min_words'] <= len(palabras) <= self.config['max_words']:
                validas += 1
        return validas / len(oraciones)

    def _check_repetitions(self, text: str) -> float:
        palabras = text.split()
        repeticiones = 0
        
        for i in range(len(palabras) - self.config['repetition_window'] + 1):
            ventana = palabras[i:i+self.config['repetition_window']]
            if len(set(ventana)) == 1:
                repeticiones += 1
                
        return 1.0 if repeticiones <= self.config['max_repetitions'] else 0.0

    def _check_suspicious_chars(self, text: str) -> float:
        caracteres_invalidos = re.findall(
            f'[^{self.config["allowed_chars"]}]', 
            text
        )
        return 1.0 - (len(caracteres_invalidos) / len(text)) if text else 1.0

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

    def __init__(self, oraciones_min: int = 3, palabras_min: int = 2, fuzzy_threshold=0.8) -> None:
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

        self.fuzzy_dedup = FuzzyDeduplicator(threshold=fuzzy_threshold)
        self.metadata = {}

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
            (r'ñ', 'ñ'),  # Mantener ñ original
            (r'Ã¡', 'á'), (r'Ã©', 'é'), (r'Ã­', 'í'),
            (r'Ã³', 'ó'), (r'Ãº', 'ú'),  # Corregir caracteres latin-1
            (r'ã', 'ã'), (r'ẽ', 'ẽ'), (r'ĩ', 'ĩ'),
            (r'õ', 'õ'), (r'ũ', 'ũ'),  # Mantener caracteres guaraníes
            (r'ĝ', 'g̃'),  # Usar la combinación unicode correcta
            (r'[“”]', '"'), (r'[‘’]', "'")
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
        directorio_salida: str,
        tolerancia: float,
        existing_hashes: set = None
    ) -> None:
        """Procesa un archivo completo y guarda el resultado.

        Args:
            ruta_entrada: Ruta al archivo de entrada
            directorio_salida: Directorio para archivos procesados

        Raises:
            IOError: Si hay problemas al leer/escribir archivos
        """

        if existing_hashes is None:
            existing_hashes = set()

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

            validador = ValidatorGuarani(tolerancia)
            # Validar contenido        
            es_valido, porcentaje, _ = validador.validate_content(contenido_limpio)

            if es_valido:
                print(f"  ✓ Válido ({porcentaje:.0%} guaraní)")


                quality = ValidadorCalidad()          
                calidad_ok, calidad_score = quality.validate_document(contenido_limpio)
                print(f"  ✓ Calidad: {calidad_score:.0%}")
                
                if not calidad_ok:
                    print(f"  ✗ Calidad insuficiente ({calidad_score:.0%} < {porcentaje})")
                    return
                
                # Verificación de duplicados
                contenido_bytes = contenido_limpio.encode('utf-8')
                file_hash = hashlib.md5(contenido_bytes).hexdigest()
                if file_hash in existing_hashes:
                    print(f"  ✗ Duplicado exacto detectado, omitiendo")
                    return
                existing_hashes.add(file_hash)

                # Deduplicación difusa
                doc_id = Path(ruta_entrada).stem
                cluster_id, is_new = self.fuzzy_dedup.process_document(doc_id, contenido_limpio)
                
                if not is_new:
                    print(f"  ✗ Duplicado aproximado detectado (Cluster {cluster_id}, Tamaño: {self.fuzzy_dedup.clusters[cluster_id]['size']})")
                    return

                # Guardar metadata
                self.metadata[doc_id] = {
                    'cluster_id': cluster_id,
                    'cluster_size': self.fuzzy_dedup.clusters[cluster_id]['size'],
                    'original_path': ruta_entrada,
                    'hash_md5': file_hash
                }

                os.makedirs(directorio_salida, exist_ok=True)
                nombre_archivo = Path(ruta_entrada).stem
                ruta_salida = os.path.join(
                    directorio_salida, 
                    f"{nombre_archivo}.txt"
                )
            
                with open(ruta_salida, 'w', encoding='utf-8') as f:
                    success = f.write(contenido_limpio)
                
                    if success:
                        print(f"  ✓ Procesado → {ruta_salida}")
                    else:
                        print("  ✗ Error en procesamiento posterior")

                metadata_path = os.path.join(directorio_salida, f"{nombre_archivo}.meta.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata[doc_id], f)

            else:
                print(f"  ✗ Inválido ({porcentaje:.0%} guaraní)")

        else:
            print(f"Documento no cumple los criterios: {ruta_entrada}")

class FuzzyDeduplicator:
    """Clase para detección de documentos similares usando técnicas de hashing aproximado.
    
    Implementa un sistema de deduplicación difusa utilizando:
    - MinHash: Para crear huellas digitales compactas de documentos
    - LSH (Local Sensitive Hashing): Para agrupar eficientemente documentos similares

    Atributos:
        threshold (float): Umbral de similitud para considerar documentos como duplicados (0.0-1.0)
        num_perm (int): Número de permutaciones para MinHash (precisión/rendimiento)
        lsh (MinHashLSH): Índice para búsqueda aproximada de vecinos cercanos
        clusters (dict): Diccionario con la metadata de los grupos de documentos similares
        next_cluster_id (int): Contador para ID de nuevos clusters

    Ejemplo de uso:
        >>> dedup = FuzzyDeduplicator(threshold=0.8)
        >>> cluster_id, es_nuevo = dedup.process_document("doc1", texto_largo)
    """

    def __init__(self, threshold=0.8, num_perm=128):
        """Inicializa el deduplicador difuso.
        
        Args:
            threshold (float, opcional): Similitud mínima para agrupar documentos. Default=0.8
            num_perm (int, opcional): Número de funciones hash para MinHash. Default=128
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.clusters = {}  # {doc_id: (minhash, metadata)}
        self.next_cluster_id = 1

    def _create_minhash(self, text):
        """Crea la firma MinHash para un documento de texto.
        
        Args:
            text (str): Contenido textual del documento
            
        Returns:
            MinHash: Firma digital del documento
            
        Nota:
            - Divide el texto en tokens por espacios
            - Usa codificación UTF-8 para los hashes
        """
        tokens = text.split()
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf-8'))
        return m

    def process_document(self, doc_id, text):
        """Procesa un documento para detectar duplicados aproximados.
        
        Args:
            doc_id (str): Identificador único del documento
            text (str): Contenido textual a procesar
            
        Returns:
            Tuple(str, bool): (ID del cluster, Es nuevo)
            
        Flujo de trabajo:
            1. Genera el MinHash del documento
            2. Consulta en el índice LSH
            3. Si existe match:
                - Incrementa el contador del cluster existente
                - Retorna (cluster_id, False)
            4. Si no existe match:
                - Crea nuevo cluster
                - Almacena metadata
                - Retorna (cluster_id, True)
        """
        minhash = self._create_minhash(text)
        results = self.lsh.query(minhash)
        
        if results:
            cluster_id = results[0]
            self.clusters[cluster_id]['size'] += 1
            return cluster_id, False  # (cluster_id, is_new)
        else:
            cluster_id = f"cluster_{self.next_cluster_id}"
            self.lsh.insert(cluster_id, minhash)
            self.clusters[cluster_id] = {
                'minhash': minhash,
                'size': 1,
                'representative': doc_id
            }
            self.next_cluster_id += 1
            return cluster_id, True

def main() -> None:
    """Función principal para procesamiento de archivos
    """
    parser = argparse.ArgumentParser(
        description='Procesamiento de corpus en guaraní'
    )

    parser.add_argument('-i', '--input', required=True, 
                      help='Directorio de entrada con archivos .txt')
    parser.add_argument('-o', '--output_dir', default='corpus/', 
                      help='Directorio de salida')
    parser.add_argument('--threshold', type=float, default=0.8,
                    help='%% mínimo de oraciones en guaraní (0-1)') 
    
    args = parser.parse_args()

    print(args.threshold)
    
    # Validar directorio de entrada
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"✗ Error: '{args.input}' no es un directorio válido")
        exit(1)
    
    # Crear directorio de salida si no existe
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar hashes existentes para deduplicación
    existing_hashes = set()
    for existing_file in output_dir.glob("*.txt"):
        try:
            with open(existing_file, "r", encoding="utf-8") as f:
                contenido = f.read()
                file_hash = hashlib.md5(contenido.encode("utf-8")).hexdigest()
                existing_hashes.add(file_hash)
        except Exception as e:
            print(f"  ✗ Error al leer archivo existente: {existing_file} - {str(e)}")
    
    # Procesar archivos .txt
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"✗ No se encontraron archivos .txt en {input_dir}")
        exit(1)
        
    limpiador = Limpiador()

    print(f"\nProcesando {len(txt_files)} archivos en {input_dir}...")
    
    for txt_file in txt_files:
        print(f"\n• Archivo: {txt_file.name}")       
        limpiador.procesar_archivo(str(txt_file), args.output_dir, args.threshold, existing_hashes)

    print("\n=== Archivos de Metadatos Generados ===")
    meta_files = list(output_dir.glob("*.meta.json"))
    
    if not meta_files:
        print("No se generaron archivos de metadatos")
    else:
        for meta_file in meta_files:
            print(f"\n• Metadata: {meta_file.name}")
            with open(meta_file, 'r', encoding='utf-8') as f:
                try:
                    contenido = json.load(f)
                    print(json.dumps(contenido, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"  ✗ Error leyendo metadata: {str(e)}")


if __name__ == "__main__":
    main()
