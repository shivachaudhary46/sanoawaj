# Importing all libraries 
from flask import Flask, jsonify, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("home.html")

if __name__== "__main__":
    app.run(debug=True, host="0.0.0.0",  port=5050)