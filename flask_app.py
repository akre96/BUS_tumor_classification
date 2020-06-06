import os, sys
from flask import Flask, render_template, request, url_for, session
from PIL import Image
from classification_model.classification import predict
from segmentation_model.predict import get_mask


app = Flask(__name__)
app.secret_key = 'TEMPORARY'
if app.secret_key == 'TEMPORARY' and os.environ['FLASK_ENV'] != 'development':
    raise ValueError('Secret Key used not for production')
if app.secret_key == 'TEMPORARY':
    print('\nWARNING SECRET KEY UNSECURE\n')

demo_img = 'static/demo_image.jpg'
demo_mask = 'static/demo_image_mask.jpg'

mask_model_path = 'segmentation_model/checkpoints/Resnet18_UNET_0.19.pth'

@app.route('/', methods=['GET', 'POST'])
def home():
    image = None
    image_name = None
    if request.method == 'POST':
        if request.form['use_demo'] == '1':
            image = url_for('static', filename='demo_image.jpg')
            image_name = 'demo_image'
            session['image_name'] = image_name
            session['image'] = image
            session['demo'] = True

    return render_template('home.html', image=image, image_name=image_name)


@app.route('/ai', methods=['GET', 'POST'])
def segment():
    image = None
    image_name = None
    ai_results = None
    show_ai = False
    render_mask = None
    render_mask_out = 'tmp/render_mask.png'
    mask = None
    if request.method == 'POST':
        if 'show_ai' in request.form.keys():
            show_ai = bool(int(request.form['show_ai']))
            print('SHOW_AI', request.form['show_ai'])

    if 'image' in session:
        image = session['image']
        image_name = session['image_name']

    if image is not None:
        abs_img = image[1:]  # Removes absolute path used by flask app
        run_ai = True
        if 'ai_pred_image' in session:
            if session['ai_pred_image'] == image:
                print('USING CACHED PREDICTION')
                run_ai = False
                ai_results = float(session['ai_results'])
        if run_ai:
            image_size = Image.open(abs_img).size
            mask_arr = get_mask(abs_img, mask_model_path)
            mask_im = Image.fromarray(mask_arr)
            mask_im = mask_im.resize(image_size)
            abs_mask = 'static/tmp/raw_mask.png'
            mask_im.save(abs_mask, 'PNG')
            ai_results = predict(abs_img,  abs_mask, 'classification_model/models/model.h5')
            mask_bg_to_transparent(abs_mask, 'static/'+render_mask_out)

        session['ai_pred_image'] = image
        session['ai_results'] = str(ai_results)

        render_mask = url_for('static', filename=render_mask_out)

    return render_template('home.html', image=image, image_name=image_name, ai_results=ai_results, mask=render_mask, show_ai=show_ai)


def mask_bg_to_transparent(img_path, out='static/tmp/temp_mask.png'):
    img = Image.open(img_path)
    img = img.convert("RGBA")
    img_data = img.getdata()

    transparent = (0, 0, 0, 0)
    red = (231, 76, 60, 255)

    bg_transp = []
    for pixel in img_data:
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            bg_transp.append(transparent)
        elif pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
            bg_transp.append(red)
        else:
            bg_transp.append(transparent)

    img.putdata(bg_transp)
    img.save(out, "PNG")
