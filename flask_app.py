from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    image = None
    image_name = None
    if request.method == 'POST':
        if request.form['use_demo'] == '1':
            image = url_for('static', filename='demo_image.jpg')
            image_name = 'demo_image'

    return render_template('home.html', image=image, image_name=image_name)
