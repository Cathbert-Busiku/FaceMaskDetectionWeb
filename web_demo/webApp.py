#import flasks libraries
import os
import requests
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import scoped_session, sessionmaker
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, Response, send_file
from flask_socketio import SocketIO, emit
import webbrowser
from threading import Timer

#T
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

#Create a route 
@app.route('/')
def index():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
    # Timer(1,lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    # socketio.run(app)
     #app.run(debug=True)
    
  