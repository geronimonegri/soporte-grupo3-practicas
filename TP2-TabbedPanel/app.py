"""App Kivy que carga la interfaz definida en test.kv (TabbedPanel con 4 pestañas).

Instalar:  pip install kivy
Ejecutar:  python app.py

Requisito: este archivo debe estar en la MISMA carpeta que test.kv.
"""

from kivy.app import App
from kivy.lang import Builder


class DemoApp(App):
    def build(self):
        # test.kv empieza con "TabbedPanel:" (widget raíz suelto, sin nombre de
        # clase), así que lo cargamos a mano con Builder.load_file y devolvemos
        # el widget resultante como raíz de la app.
        return Builder.load_file("test.kv")


if __name__ == "__main__":
    DemoApp().run()