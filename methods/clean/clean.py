#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de Limpieza para Corpus en Guaraní - Versión 3.0

Este script implementa un pipeline completo para la limpieza y normalización de corpus de texto en guaraní,
con soporte para deduplicación global, rehidratación y generación de documentación automática.

Características principales:
- Procesamiento eficiente de grandes volúmenes de texto
- Deduplicación exacta (hash MD5) y aproximada (MinHash + LSH)
- Validación lingüística robusta con modelos FastText
- Generación automática de metadatos y documentación
- Soporte para caracteres especiales del guaraní
- Compatibilidad con procesamientos incrementales
"""

from datasketch import MinHashLSH, MinHash
import json
import re
import os
import sys
from pathlib import Path
from typing import Tuple, List, Pattern, Optional, Dict, Any
import argparse
import fasttext
import nltk
from huggingface_hub import hf_hub_download
import hashlib
from datetime import datetime
import shutil


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
        self.threshold = threshold
        self.config = {
            'max_upper_ratio': 0.2,
            'min_words': 3,
            'max_words': 50,
            'repetition_window': 5,
            'max_repetitions': 2,
            'allowed_chars': r'[a-zA-ZãẽĩõũáéíóúýñçÁÉÍÓÚÝÑÇ\'\- \t\n.,!?;:¿¡%&$#@()]'
        }
    
    def validate_document(self, text: str) -> Tuple[bool, float]:
        scores = []
        cap_score = self._check_capitalization(text)
        scores.append(cap_score * 0.2)
        len_score = self._check_sentence_lengths(text)
        scores.append(len_score * 0.3)
        rep_score = self._check_repetitions(text)
        scores.append(rep_score * 0.25)
        char_score = self._check_suspicious_chars(text)
        scores.append(char_score * 0.25)
        
        total_score = sum(scores)
        return (total_score >= self.threshold, total_score)

    def _check_capitalization(self, text: str) -> float:
        letras = [c for c in text if c.isalpha()]
        if not letras:
            return 1.0
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

    def __init__(self, oraciones_min: int = 3, palabras_min: int = 2, 
                 fuzzy_dedup: Optional[Any] = None) -> None:
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
        self.fuzzy_dedup = fuzzy_dedup
        self.metadata = {}
        self.stats = {
            'total': 0,
            'conservados': 0,
            'duplicados_exactos': 0,
            'duplicados_aproximados': 0,
            'invalidos': 0,
            'baja_calidad': 0
        }

    def es_relevante(self, texto: str) -> bool:
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
        texto = re.sub(r'<[^>]+>', ' ', texto)
        texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        reemplazos = [
            (r'ñ', 'ñ'),
            (r'Ã¡', 'á'), (r'Ã©', 'é'), (r'Ã­', 'í'),
            (r'Ã³', 'ó'), (r'Ãº', 'ú'),
            (r'ã', 'ã'), (r'ẽ', 'ẽ'), (r'ĩ', 'ĩ'),
            (r'õ', 'õ'), (r'ũ', 'ũ'),
            (r'ĝ', 'g̃'),
            (r'[“”]', '"'), (r'[‘’]', "'")
        ]
        for patron, reemplazo in reemplazos:
            texto = re.sub(patron, reemplazo, texto)
        return texto

    def limpiar_documento(self, texto: str) -> Tuple[bool, str]:
        normalizado = self.normalizar_texto(texto)
        return self.es_relevante(normalizado), normalizado

    def procesar_archivo(
        self, 
        ruta_entrada: str, 
        directorio_salida: str,
        tolerancia: float,
        global_dedup: Optional[Any] = None,
        existing_hashes: set = None
    ) -> None:
        self.stats['total'] += 1
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
                print(f"  ✗ Error al leer archivo: {str(e)}")
                return
        except Exception as e:
            print(f"  ✗ Error al leer archivo: {str(e)}")
            return
        
        se_conserva, contenido_limpio = self.limpiar_documento(contenido)
        if not se_conserva:
            print(f"  ✗ Filtrado por criterios básicos")
            return
        
        validador = ValidatorGuarani(tolerancia)
        es_valido, porcentaje, _ = validador.validate_content(contenido_limpio)
        if not es_valido:
            print(f"  ✗ Inválido ({porcentaje:.0%} guaraní)")
            self.stats['invalidos'] += 1
            return

        print(f"  ✓ Válido ({porcentaje:.0%} guaraní)")
        quality = ValidadorCalidad()          
        calidad_ok, calidad_score = quality.validate_document(contenido_limpio)
        print(f"  ✓ Calidad: {calidad_score:.0%}")
        if not calidad_ok:
            print(f"  ✗ Calidad insuficiente ({calidad_score:.0%})")
            self.stats['baja_calidad'] += 1
            return
        
        # Verificación de duplicados exactos
        contenido_bytes = contenido_limpio.encode('utf-8')
        file_hash = hashlib.md5(contenido_bytes).hexdigest()
        if file_hash in existing_hashes:
            print(f"  ✗ Duplicado exacto detectado, omitiendo")
            self.stats['duplicados_exactos'] += 1
            return
        existing_hashes.add(file_hash)

        # Deduplicación difusa
        doc_id = Path(ruta_entrada).stem
        cluster_id = None
        is_new = True
        
        if self.fuzzy_dedup:
            cluster_id, is_new = self.fuzzy_dedup.process_document(
                doc_id, 
                contenido_limpio, 
                global_dedup
            )
            if not is_new:
                print(f"  ✗ Duplicado aproximado detectado (Cluster {cluster_id})")
                self.stats['duplicados_aproximados'] += 1
                # Omitir documento si es duplicado aproximado
                return

        # Guardar metadata
        metadata_entry = {
            'original_path': ruta_entrada,
            'hash_md5': file_hash,
            'fecha_procesamiento': datetime.now().isoformat(),
            'valido_guarani': porcentaje,
            'calidad': calidad_score
        }
        
        if cluster_id and self.fuzzy_dedup:
            cluster_size = self.fuzzy_dedup.clusters.get(cluster_id, {}).get('size', 1)
            metadata_entry['cluster_id'] = cluster_id
            metadata_entry['cluster_size'] = cluster_size
            
            if global_dedup:
                global_size = global_dedup.clusters.get(cluster_id, {}).get('size', 1)
                metadata_entry['global_cluster_size'] = global_size
                metadata_entry['weight'] = 1 / global_size

        self.metadata[doc_id] = metadata_entry

        # Guardar documento procesado
        os.makedirs(directorio_salida, exist_ok=True)
        nombre_archivo = Path(ruta_entrada).stem
        ruta_salida = os.path.join(directorio_salida, f"{nombre_archivo}.txt")
        try:
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                f.write(contenido_limpio)
                print(f"  ✓ Guardado → {ruta_salida}")
                self.stats['conservados'] += 1
        except Exception as e:
            print(f"  ✗ Error al guardar archivo: {str(e)}")
            return

        # Guardar metadatos
        metadata_path = os.path.join(directorio_salida, f"{nombre_archivo}.meta.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_entry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  ✗ Error al guardar metadatos: {str(e)}")

class FuzzyDeduplicator:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.clusters = {}
        self.next_cluster_id = 1

    def _create_minhash(self, text: str) -> MinHash:
        tokens = text.split()
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf-8'))
        return m

    def process_document(
        self, 
        doc_id: str, 
        text: str, 
        global_dedup: Optional[Any] = None
    ) -> Tuple[Optional[str], bool]:
        minhash = self._create_minhash(text)
        results = self.lsh.query(minhash)
        cluster_id = None
        is_new = True
        
        if results:
            cluster_id = results[0]
            is_new = False
            if cluster_id in self.clusters:
                self.clusters[cluster_id]['size'] += 1
            else:
                self.clusters[cluster_id] = {'size': 1, 'documents': [doc_id]}
            if global_dedup:
                if cluster_id in global_dedup.clusters:
                    global_dedup.clusters[cluster_id]['size'] += 1
                else:
                    global_dedup.clusters[cluster_id] = {'size': 1}
        else:
            cluster_id = f"cluster_{self.next_cluster_id}"
            self.lsh.insert(cluster_id, minhash)
            self.clusters[cluster_id] = {
                'minhash': minhash, 
                'size': 1, 
                'documents': [doc_id],
                'representative': doc_id
            }
            self.next_cluster_id += 1
            if global_dedup:
                global_dedup.clusters[cluster_id] = {'size': 1}

        return cluster_id, is_new

class GlobalDeduplicator:
    def __init__(self, state_dir: str = "dedupe_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.clusters = self.load_clusters()
    
    def load_clusters(self) -> Dict:
        clusters_file = self.state_dir / "clusters.json"
        if clusters_file.exists():
            try:
                with open(clusters_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  ⚠ Error cargando estado: {str(e)}")
                return {}
        return {}
    
    def save_clusters(self):
        clusters_file = self.state_dir / "clusters.json"
        try:
            with open(clusters_file, "w") as f:
                json.dump(self.clusters, f, indent=2)
            print(f"  ✓ Estado guardado en {clusters_file}")
        except Exception as e:
            print(f"  ✗ Error guardando estado: {str(e)}")

def generate_readme(output_dir: str, params: Dict, stats: Dict):
    # Función auxiliar para obtener un valor de forma segura, con un valor por defecto
    def get_param(dict_obj: Dict, key: str, default):
        return dict_obj.get(key, default)

    # --- Parámetros de Ejecución ---
    threshold = get_param(params, 'threshold', 0.0)
    dedupe_mode = get_param(params, 'dedupe_mode', 'N/A')
    rehydrate = get_param(params, 'rehydrate', False)
    input_dir = get_param(params, 'input', 'N/A')
    output_dir_param = get_param(params, 'output_dir', 'N/A') # Renombrado para evitar conflicto con output_dir de la función
    dedupe_state = get_param(params, 'dedupe_state', 'N/A')

    # --- Estadísticas de Procesamiento ---
    total = get_param(stats, 'total', 0)
    conservados = get_param(stats, 'conservados', 0)
    duplicados_exactos = get_param(stats, 'duplicados_exactos', 0)
    duplicados_aproximados = get_param(stats, 'duplicados_aproximados', 0)
    invalidos = get_param(stats, 'invalidos', 0)
    baja_calidad = get_param(stats, 'baja_calidad', 0)
    clusters = get_param(stats, 'clusters', 0)

    # Cálculo seguro de porcentaje
    conservados_porcentaje = (conservados / total) if total > 0 else 0


    content = f"""
    # Corpus Guaraní - Procesamiento
    **Fecha**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        
    ## Parámetros de Ejecución
    - Umbral validación guaraní: {threshold:.0%}
    - Modo deduplicación: {dedupe_mode}
    - Rehidratación: {'Activada' if rehydrate else 'Desactivada'}
    
    - Directorio entrada: {input_dir}
    - Directorio salida: {output_dir_param}

    ## Estadísticas de Procesamiento
    - Documentos procesados: {total}
    - Documentos conservados: {conservados} ({conservados_porcentaje:.0%})
    - Duplicados exactos: {duplicados_exactos}
    - Duplicados aproximados: {duplicados_aproximados}
    - Inválidos (no guaraní): {invalidos}
    - Rechazados por calidad: {baja_calidad}
    - Clusters detectados: {clusters}

    ## Transformaciones Aplicadas
    1. Filtrado por longitud y contenido irrelevante
    2. Normalización Unicode y corrección de caracteres
    3. Validación lingüística (guaraní)
    4. Evaluación de calidad de texto
    5. Deduplicación ({dedupe_mode})
    6. {'Ponderación por tamaño de cluster (rehidratación)' if rehydrate else ''}

    ## Estructura de Archivos
    - Archivos de texto: `*.txt`
    - Metadatos: `*.meta.json`
    - Documentación: `README.md`
    - Estado deduplicación: `{dedupe_state}/clusters.json`

    ## Reproducibilidad
    Para reproducir este procesamiento:
    ```bash
    ./clean.sh {input_dir} {output_dir_param} \\
    --threshold={threshold} \\
    --dedupe-mode={dedupe_mode} \\
    --dedupe-state={dedupe_state} \\
    {"--rehydrate" if rehydrate else ""}
    """
    readme_path = Path(output_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f" ✓ README generado en {readme_path}")

def main() -> None:
    """Función principal para procesamiento de archivos
    """
    parser = argparse.ArgumentParser(
        description='Procesamiento de corpus en guaraní'
    )

    parser.add_argument('-i', '--input', required=True, 
                    help='Directorio de entrada con archivos .txt')
    parser.add_argument('-o', '--output_dir', default='corpus/', 
                        help='Directorio de salida para archivos procesados')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Porcentaje mínimo de oraciones en guaraní (0-1)') 
    parser.add_argument('--dedupe-mode', choices=['global', 'local', 'none'], default='global',
                        help='Modo de deduplicación: global (persistente), local (memoria), none (sin deduplicación)')
    parser.add_argument('--dedupe-state', default='dedupe_state',
                        help='Directorio para almacenar estado de deduplicación global')
    parser.add_argument('--rehydrate', action='store_true',
                        help='Activar ponderación por tamaño de cluster (rehidratación)')
    parser.add_argument('--clear-state', action='store_true',
                        help='Limpiar estado de deduplicación antes de comenzar')

    args = parser.parse_args()

    # Validar directorio de entrada
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"✗ Error: '{args.input}' no es un directorio válido")
        sys.exit(1)

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manejar estado de deduplicación
    global_dedup = None
    if args.dedupe_mode == 'global':
        state_dir = Path(args.dedupe_state)
        
        if args.clear_state and state_dir.exists():
            try:
                shutil.rmtree(state_dir)
                print(f"  ✓ Estado limpiado: {state_dir}")
            except Exception as e:
                print(f"  ✗ Error limpiando estado: {str(e)}")
        
        global_dedup = GlobalDeduplicator(args.dedupe_state)

    # Inicializar deduplicador fuzzy
    fuzzy_dedup = None
    if args.dedupe_mode != 'none':
        fuzzy_dedup = FuzzyDeduplicator(threshold=0.8)
        
    # Inicializar limpiador con deduplicador
    limpiador = Limpiador(fuzzy_dedup=fuzzy_dedup)

    # Cargar hashes existentes para deduplicación exacta
    existing_hashes = set()
    for existing_file in output_dir.glob("*.txt"):
        try:
            with open(existing_file, "r", encoding="utf-8") as f:
                contenido = f.read()
                file_hash = hashlib.md5(contenido.encode("utf-8")).hexdigest()
                existing_hashes.add(file_hash)
        except Exception as e:
            print(f"  ✗ Error al leer archivo existente: {existing_file} - {str(e)}")

    # Procesar archivos
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"✗ No se encontraron archivos .txt en {input_dir}")
        sys.exit(1)
        
    print(f"\n▶ Procesando {len(txt_files)} archivos en {input_dir}...")
    print(f"  Modo deduplicación: {args.dedupe_mode}")
    print(f"  Umbral guaraní: {args.threshold:.0%}")
    print(f"  Rehidratación: {'Activada' if args.rehydrate else 'Desactivada'}")

    kept_count = 0
    
    # Procesar cada archivo
    for i, txt_file in enumerate(txt_files):
        print(f"\n▷ [{i+1}/{len(txt_files)}] Archivo: {txt_file.name}")
        limpiador.procesar_archivo(
            str(txt_file), 
            str(output_dir),
            args.threshold,
            global_dedup,
            existing_hashes
        )
        kept_count += 1

    # Guardar estado global si es necesario
    if global_dedup:
        global_dedup.save_clusters()

    # Preparar estadísticas para README
    stats = limpiador.stats.copy()
    if fuzzy_dedup:
        stats['clusters'] = len(fuzzy_dedup.clusters)
    else:
        stats['clusters'] = 0
        print(f"  ✗ Error leyendo metadata")

   # Generar README
    generate_readme(args.output_dir, {
        'threshold': args.threshold,
        'dedupe_mode': args.dedupe_mode,
        'rehydrate': args.rehydrate
    }, {
        'total': len(txt_files),
        'kept': kept_count,
        'clusters': len(limpiador.fuzzy_dedup.clusters) if fuzzy_dedup else 0
    })

    # Mostrar metadatos
    print("\n=== Archivos de Metadatos Generados ===")
    meta_files = list(output_dir.glob("*.meta.json"))
    if meta_files:
        for meta_file in meta_files:
            print(f"\n• Metadata: {meta_file.name}")
            with open(meta_file, 'r', encoding='utf-8') as f:
                try:
                    contenido = json.load(f)
                    print(json.dumps(contenido, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"  ✗ Error leyendo metadata: {str(e)}")
    else:
        print("No se generaron archivos de metadatos")



if __name__ == "__main__":
    main()
