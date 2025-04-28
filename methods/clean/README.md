
- Probado en python3.9


methods/clean.py/
│
├── clean.py 
├── requirements.txt 
└── README.md  


Documentación
Visión General
Este script implementa el pipeline de pre-filtrado y normalización para corpus en guaraní según los requerimientos del proyecto UC Autumn of Code 2025. El pipeline:

Realiza filtrado básico eliminando:
- Documentos con menos de 3 oraciones
- Líneas con menos de 2 palabras
- Documentos que contengan patrones irrelevantes ("lorem ipsum", "javascript", etc.)

Normaliza el texto mediante:
- Eliminación de etiquetas HTML
- Corrección de problemas de codificación
- Estandarización de espacios
- Normalización de caracteres específicos del guaraní (como g̃ → ĝ)


Uso
Instalar dependencias:

pip install -r requirements.txt

En la raiz del proyecto:
- Dar permiso de ejecucion al archivo clean.sh
  sudo chmod +x clean.sh 

Ejecutar el limpiador desde la raiz del proyecto:
./clean.sh test/samples/gn.txt carpeta_salida_opcional (si se omite carpeta_salida_opcional la salida se guarda en la carpeta corpus)

Resultado: 
El resultado limpio se guardará en el directorio especificado con "_limpio" añadido al nombre del archivo.


Convenciones PEP 8 Implementadas

Nombres:
- Clases: PascalCase (LimpiadorGuarani)
- Métodos/Funciones: snake_case (normalizar_texto)
- Variables: snake_case (patrones_irrelevantes)
- Constantes: MAYÚSCULAS (PATRONES_IRRELEVANTES)

Formato:
- Líneas limitadas a 79 caracteres
- Sangrado de 4 espacios
- Espacios alrededor de operadores y después de comas
- Líneas en blanco entre métodos/clases

Tipado:
- Uso de type hints en todas las funciones/métodos
- Comentarios de tipo para atributos de clase

Documentación:
- Docstrings en formato Google para todos los módulos, clases y métodos
- Comentarios explicativos para secciones complejas