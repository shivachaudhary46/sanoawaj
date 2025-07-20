# Importing all libraries 
from flask import Flask, jsonify, request, render_template, redirect, url_for

# intialising app, Flask name 
app = Flask(__name__)

# 
@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("home.html")

'''
running app if an only if file name is equal to app.py 
'''
if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0",  port=5050)