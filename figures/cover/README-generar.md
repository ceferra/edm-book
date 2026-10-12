# Portadas: evaluación y Machine Learning
Tres composiciones abstractas originales, realizadas mediante código, sin texto.
Son reinterpretaciones nuevas de las direcciones estéticas propuestas; no reproducen las imágenes anteriores.

## Uso
Python 3. Para exportar también PNG: `pip install -r requirements.txt`.
Desde esta carpeta: `python generar.py --seed 42 --out resultados`.
Cambiar la semilla genera variaciones (especialmente geometría y flujo).

01_geometria: retícula de cuadriláteros deformada localmente, marfil/negro/rojo.
02_flujo: integración de trayectorias de un campo vectorial suave, azul noche/turquesa/naranja.
03_lineas: familias de curvas sinusoidales entrelazadas, blanco y colores vibrantes.

Formato 1600 × 2400 (2:3), SVG vectorial y PNG. Espacio superior para título.
Los colores, densidades y deformaciones se ajustan en generar.py.
El SVG permite adaptar la resolución sin perder calidad. No incluyen lomo, contraportada ni sangrado: requieren las medidas finales de imprenta.
PNG para previsualización; para imprimir, usar el SVG y gestionar color con la imprenta.
