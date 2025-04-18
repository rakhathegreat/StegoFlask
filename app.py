#------ External Libraries ------#
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import bitstring
import numpy as np
import zigzag as zz
import os
import io
#================================#
#---------- Source Files --------#
import image_preparation as img
import data_embedding as stego
#================================#
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
NUM_CHANNELS = 3

def image_to_bytes(image):
    is_success, buffer = cv2.imencode(".png", image)
    if not is_success:
        raise ValueError("Could not encode image")
    return io.BytesIO(buffer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['GET', 'POST'])
def embed():
    stego_image_filename = None
    if request.method == 'POST':
        if 'file' not in request.files or 'message' not in request.form:
            flash('Harap upload gambar dan masukkan pesan')
            return redirect(request.url)
        
        file = request.files['file']
        message = request.form['message']
        
        if file.filename == '' or message == '':
            flash('File atau pesan tidak valid')
            return redirect(request.url)
        
        try:
            # Read image
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            file_bytes = in_memory_file.getvalue()
            file_np_arr = np.frombuffer(file_bytes, np.uint8)
            raw_cover_image = cv2.imdecode(file_np_arr, cv2.IMREAD_COLOR)

            # Process image dimensions
            height, width = raw_cover_image.shape[:2]
            while height % 8: 
                height += 1
            while width % 8: 
                width += 1
            valid_dim = (width, height)

            padded_image = cv2.resize(raw_cover_image, valid_dim)
            cover_image_f32 = np.float32(padded_image)
            cover_image_YCC = img.YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

            stego_image = np.empty_like(cover_image_f32)

            # Embedding process
            for chan_index in range(NUM_CHANNELS):
                dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]
                dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]
                sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

                if chan_index == 0:
                    secret_data = ""
                    for char in message.encode('ascii'):
                        secret_data += bitstring.pack('uint:8', char)
                    embedded_dct_blocks = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                    desorted_coefficients = [zz.inverse_zigzag(block, 8, 8) for block in embedded_dct_blocks]
                else:
                    desorted_coefficients = [zz.inverse_zigzag(block, 8, 8) for block in sorted_coefficients]

                dct_dequants = [np.multiply(data, img.JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]
                idct_blocks = [cv2.idct(block) for block in dct_dequants]
                stego_image[:, :, chan_index] = np.asarray(
                    img.stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

            # Post-processing
            stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)
            final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))

            # flash("Pesan berhasil disematkan!", "success")
            # # Return result
            # img_bytes = image_to_bytes(final_stego_image)
            # return send_file(
            #     img_bytes,
            #     mimetype='image/png',
            #     as_attachment=True,
            #     download_name='stego_image.png'
            # )
            
            # Simpan ke folder static/uploads
            filename = secure_filename("stego_image.png")
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(save_path, final_stego_image)

            flash("Pesan berhasil disematkan!", "success")
            stego_image_filename = filename
        
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('embed.html', stego_image=stego_image_filename)

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Harap upload gambar stego')
            return redirect(request.url)
        
        file = request.files['file']
        
        try:
            # Read image
            in_memory_file = io.BytesIO()
            file.save(in_memory_file)
            file_bytes = in_memory_file.getvalue()
            file_np_arr = np.frombuffer(file_bytes, np.uint8)
            stego_image = cv2.imdecode(file_np_arr, cv2.IMREAD_COLOR)

            # Extraction process
            stego_image_f32 = np.float32(stego_image)
            stego_image_YCC = img.YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

            dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]
            dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]
            sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

            recovered_data = stego.extract_encoded_data_from_DCT(sorted_coefficients)
            recovered_data.pos = 0

            # Read message
            data_len = int(recovered_data.read('uint:32')) // 8  # Perbaikan di sini
            extracted_data = bytearray()

            for _ in range(data_len):
                extracted_data.append(recovered_data.read('uint:8'))

            secret_message = extracted_data.rstrip(b'\x00').decode('ascii')
            return render_template('extract_result.html', message=secret_message)
            
        except Exception as e:
            flash('Gagal mengekstrak pesan. Mungkin gambar tidak mengandung pesan atau korup.')
            return redirect(request.url)
    
    return render_template('extract.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)