from flask import Flask, render_template, request, url_for
from classification_model.classification import predict

app = Flask(__name__)

demo_mask = '/data/dataset_busi/malignant/malignant (16)_mask.png'
demo_img = '/data/dataset_busi/malignant/malignant (16).png'
pred = predict(demo_img, demo_mask, 'classification_model/models/model.h5')
print(pred)

@app.route('/', methods=['GET', 'POST'])
def home():
    image = None
    image_name = None
    if request.method == 'POST':
        if request.form['use_demo'] == '1':
            image = url_for('static', filename='demo_image.jpg')
            image_name = 'demo_image'

    return render_template('home.html', image=image, image_name=image_name)


@app.route('/ai', methods=['GET', 'POST'])
def segment():
    return render_template('home.html')
