""" Flask Web App for BUS Lesion Segmentation and Diagnosis

Run the dockerfile or after installing requirements run `strat_flask_dev.sh` to start app.

Author: Samir Akre
"""
import os
from pathlib import Path
from flask import (
    Flask,
    render_template,
    request,
    url_for,
    session,
)
from werkzeug.utils import secure_filename
from PIL import Image
from classification_model.classification import predict
from segmentation_model.predict import get_mask


app = Flask(__name__)
app.secret_key = os.environ['FLASK_SECRET']
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
mask_model_path = 'segmentation_model/checkpoints/Resnet18_UNET_0.19.pth'

if not Path('static/tmp').is_dir():
    print('Making folder at static/tmp')
    Path('static/tmp').mkdir()

@app.route('/', methods=['GET', 'POST'])
def home():
    """ Returns home page to flask app

    Returns:
        HTML Render: HTML rendered Jinja2 Template of Home
    """
    image = None
    image_name = None

    # If form submit from `Use Demo Image` or `Use Uploaded Image` buttons
    if request.method == 'POST':
        # If demo, return demo image
        if 'use_demo' in request.form:
            image = url_for('static', filename='demo_image.jpg')
            image_name = 'demo_image'
            session['demo'] = True
        # If user uploaded file, load user image
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
        
        # Save to session for app universal use
        session['image_name'] = image_name
        session['image'] = image

    return render_template('home.html', image=image, image_name=image_name)


@app.route('/ai', methods=['GET', 'POST'])
def segment():
    """ Segment and diagnose BUS image

    Returns:
        HTML Render: Rendered Results page
    """
    image = None
    image_name = None
    ai_results = None
    show_ai = False
    render_mask = None
    render_mask_out = 'tmp/render_mask.png'

    # If segment and diagnose image button clicked
    if request.method == 'POST':
        if 'show_ai' in request.form.keys():
            show_ai = bool(int(request.form['show_ai']))
            print('SHOW_AI', request.form['show_ai'])

    # Get path to previously used image
    if 'image' in session:
        image = session['image']
        image_name = session['image_name']

    if image is not None:
        abs_img = image[1:]  # Removes absolute path used by flask app
        run_ai = True
        # Check if image already processed
        if 'ai_pred_image' in session:
            if session['ai_pred_image'] == image:
                print('USING CACHED PREDICTION')
                run_ai = False
                ai_results = session['ai_results']

        # Segment and assign risk to lesions in image
        if run_ai:
            image_size = Image.open(abs_img).size

            # Get lesion mask and resize from 256x256 back to original image size
            mask_arr = get_mask(abs_img, mask_model_path)
            mask_im = Image.fromarray(mask_arr)
            mask_im = mask_im.resize(image_size)
            abs_mask = 'static/tmp/raw_mask.png'
            mask_im.save(abs_mask, 'PNG')

            # Predict lesion malignancy
            ai_results = predict(abs_img,  abs_mask, 'classification_model/models/model.h5')

            # Creates transparent background version for display
            mask_bg_to_transparent(abs_mask, 'static/'+render_mask_out)

        ai_results = round(float(str(ai_results)), 6)
        session['ai_pred_image'] = image
        session['ai_results'] = ai_results

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
    """ Converts binary mask to RGBA where 0 is transparent

    Args:
        img_path (str): Path to mask
        out (str, optional): Save path. Defaults to 'static/tmp/temp_mask.png'.
    """
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
