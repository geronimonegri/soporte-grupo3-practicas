from flask import Flask, render_template, request, redirect
import sqlite3

app = Flask(__name__)


def crear_bd():
    cone = sqlite3.connect("peliculas.db")
    cursor = cone.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS peliculas(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            descripcion TEXT NOT NULL
        )
    """)

    cone.commit()
    cone.close()



@app.route("/")
@app.route("/peliculas")
def listar_pelis():

    conexion = sqlite3.connect("peliculas.db")
    cursor = conexion.cursor()

    cursor.execute("SELECT * FROM peliculas")
    peliculas = cursor.fetchall()

    conexion.close()

    return render_template("index.html", peliculas=peliculas)



@app.route("/nueva", methods=["GET", "POST"])
def nueva_peli():

    if request.method == "POST":

        titulo = request.form["titulo"]
        descripcion = request.form["descripcion"]

        conexion = sqlite3.connect("peliculas.db")
        cursor = conexion.cursor()

        cursor.execute(
            "INSERT INTO peliculas(titulo, descripcion) VALUES (?, ?)",
            (titulo, descripcion)
        )

        conexion.commit()
        conexion.close()

        return redirect("/peliculas")

    return render_template("nuevapeli.html")



@app.route("/editar/<int:id>", methods=["GET", "POST"])
def editar_peli(id):

    conexion = sqlite3.connect("peliculas.db")
    cursor = conexion.cursor()

    if request.method == "POST":

        titulo = request.form["titulo"]
        descripcion = request.form["descripcion"]

        cursor.execute(
            """
            UPDATE peliculas
            SET titulo=?, descripcion=?
            WHERE id=?
            """,
            (titulo, descripcion, id)
        )

        conexion.commit()
        conexion.close()

        return redirect("/peliculas")

    cursor.execute(
        "SELECT * FROM peliculas WHERE id=?",
        (id,) #espera una tupla
    )

    pelicula = cursor.fetchone()

    conexion.close()

    return render_template(
        "editarpeli.html",
        pelicula=pelicula
    )


@app.route("/eliminar/<int:id>")
def eliminar_pelicula(id):

    conexion = sqlite3.connect("peliculas.db")
    cursor = conexion.cursor()

    cursor.execute(
        "DELETE FROM peliculas WHERE id=?",
        (id,)
    )

    conexion.commit()
    conexion.close()

    return redirect("/peliculas")


if __name__ == "__main__":
    crear_bd()
    app.run(debug=True)