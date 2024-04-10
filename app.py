# libraries
import os 
from flask import Flask, request, send_file, jsonify
from english import video_to_text_prediction_english, text_to_video_prediction_english
from arabic import video_to_text_prediction_arabic, text_to_video_prediction_arabic


# define flask app
app = Flask(__name__)


############################################ video to text in english 
@app.route("/video_to_text_english", methods=['POST'])
def video_to_text_english():

    # upload file
    # this is the video itself
    video = request.files['video']
    file_name = video.filename.split("/")[-1]

    # make a fixed video name to make overrite and make sure that the uploaded file just have only 1 video file
    file_name = "new_video.mp4"

    print("this is the path of the video", file_name)

    # save the video -> to access it and make the detection process
    # after saving the video we will get the 1st parameter for the detection method -> [video_path]
    video.save(os.path.join('uploaded', file_name))

    # apply model on it
    output = video_to_text_prediction_english(os.path.join('uploaded', file_name))

    return jsonify([{"text":output}])



########################################## text to video in english
@app.route("/text_to_video_english", methods=['POST'])
def text_to_video_english():

    # request the text from the client as form data
    # text_value = request.form.get('text')
    
    # request the text as raw data 
    text_value = str(request.data ,encoding='utf-8')
    
    print(type(text_value))
    print(text_value)
    

    # apply model on the text
    output_video_path = text_to_video_prediction_english(text_value)
    print(output_video_path)
    
    return send_file(output_video_path)


########################################## video to text in arabic 
@app.route("/video_to_text_arabic", methods=['POST'])
def video_to_text_arabic():

    # upload file
    # this is the video itself
    video = request.files['video']
    file_name = video.filename.split("/")[-1]
    
    # make a fixed video name to make overrite and make sure that the uploaded file just have only 1 video file
    file_name = "new_video.mp4"
    
    print("this is the path of the video", file_name)

    # save the video -> to access it and make the detection process
    # after saving the video we will get the 1st parameter for the detection method -> [video_path]
    
   # save the new video
    video.save(os.path.join('uploaded', file_name))
        
    # apply model on it
    output = video_to_text_prediction_arabic(os.path.join('uploaded', file_name))
    
    return jsonify([{"text":output}])


########################################## text to video in arabic
@app.route("/text_to_video_arabic", methods=['POST'])
def text_to_video_arabic():

    # request the text from the client as form data
    # text_value = request.form.get('text')
    
    # request the text as raw data 
    text_value = str(request.data ,encoding='utf-8')
    
    print(type(text_value))
    print(text_value)
    

    # apply model on the text
    output_video_path = text_to_video_prediction_arabic(text_value)
    print(output_video_path)
    
    return send_file(output_video_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5000)