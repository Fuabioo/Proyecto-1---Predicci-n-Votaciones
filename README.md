Inteligencia Artificial
======

Grupo #8
------
- Fabio Mora Cubillo - 2013012801
- Sergio Moya Valerin - 2013015682
- Gabriel Venegas Castro - 2013115967

Proyectos
------
- Proyecto Corto 1 - Simulador Votos
- Proyecto 1 - Predicción Votaciones

Instalacion
------
- **Opcion #1 (Preferible):**
```bash
python setup.py sdist
python -m pip install dist/tec-2.X.tar.gz
```
- **Opcion #2:**
```bash
python setup.py install
```

Pruebas:
------
En la carpeta test se encuentran archivos de prueba:
- **g08_pytest.py**: contiene las pruebas unitarias realizadas en *pytest*
- **g08_graficos.py**: contiene la generacion de graficos
  > Los graficos incluyen muestras de 1000, 10000 y 100000 para pais y tambien provincias.

Argumentos:
------
El modulo incluye dos archivos:
- **g08_indicadores.py**: que incluye los indicadores que obtuvimos en formato csv
- **g08_actas_final.py**: que incluye los actas que obtuvimos en formato csv
Sin embargo si se quiere especificar el archivo se pueden usar los argumentos:
  > --indicadores <archivo.csv>
--actas <archivo.csv>

   Ambos son obviables, puede especificarse uno, el otro, los dos o ninguno