import os, sys
from pathlib import Path
from flask import (
    Flask,
    render_template,
    request,
    url_for,
    session,
    send_from_directory
)
from werkzeug.utils import secure_filename
from PIL import Image
from classification_model.classification import predict
from segmentation_model.predict import get_mask


app = Flask(__name__)
app.secret_key = os.environ['FLASK_SECRET']
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
mask_model_path = 'segmentation_model/checkpoints/Resnet18_UNet_0.19.pth'

if not Path('static/tmp').is_dir():
    print('Making folder at static/tmp')
    Path('static/tmp').mkdir()

@app.route('/', methods=['GET', 'POST'])
def home():
    image = None
    image_name = None
    if request.method == 'POST':
        if 'use_demo' in request.form:
            image = url_for('static', filename='demo_image.jpg')
            image_name = 'demo_image'
            session['demo'] = True
        elif 'file_upload' in request.form:
            if 'file' in request.files:
                up_file = request.files['file']
                if up_file:
                    fname = secure_filename(up_file.filename)
                    up_file.save(os.path.join('static/tmp', fname))

                    image = url_for('static', filename=os.path.join('tmp', fname))
                    image_name = '.'.join(fname.split('.')[:-1])
        else:
            print(request.form)
        session['image_name'] = image_name
        session['image'] = image

    return render_template('home.html', image=image, image_name=image_name)


@app.route('/ai', methods=['GET', 'POST'])
def segment():
    image = None
    image_name = None
    ai_results = None
    show_ai = False
    render_mask = None
    render_mask_out = 'tmp/render_mask.png'
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
                ai_results = round(float(session['ai_results']), 6)
        if run_ai:
            image_size = Image.open(abs_img).size
            mask_arr = get_mask(abs_img, mask_model_path)
            mask_im = Image.fromarray(mask_arr)
            mask_im = mask_im.resize(image_size)
            abs_mask = 'static/tmp/raw_mask.png'
            mask_im.save(abs_mask, 'PNG')
            ai_results = round(
                predict(abs_img,  abs_mask, 'classification_model/models/model.h5'),
                6
            )
            mask_bg_to_transparent(abs_mask, 'static/'+render_mask_out)

        session['ai_pred_image'] = image
        session['ai_results'] = str(ai_results)

        render_mask = url_for('static', filename=render_mask_out)

    return render_template(
        'home.html',
        image=image,
        image_name=image_name,
        ai_results=ai_results,
        mask=render_mask,
        show_ai=show_ai
    )


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
