from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty

class Calculadora(GridLayout):
    display = StringProperty("")

    def agregar(self, valor):
        self.display += str(valor)

    def calcular(self):
        try:
            self.display = str(eval(self.display))
        except:
            self.display = "Error"

    def limpiar(self):
        self.display = ""

class TestApp(App):
    def build(self):
        return Calculadora()

TestApp().run()