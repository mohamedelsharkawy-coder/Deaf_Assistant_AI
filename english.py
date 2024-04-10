##################### libraries
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import string 
import bz2file as bz2

########################################  video to text  ####################################################################################

##################### load google model 
# create object just focus on the hands 
mp_hands = mp.solutions.hands

# get the model that detect hand_landmarks 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

######################## load my model --> the original model file with no compression
# model_file = open('english_model.pickle','rb')
# model_dict = pickle.load(model_file)
# model = model_dict['model']
# model_file.close()

########################### define a function that decompressed the compressed file 
########## load my model --> the compressed model
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

model = decompress_pickle("compressed_english_model.pbz2")

######################## label map dictionary
index_to_class = {0: 'A', 1: 'About', 2: 'After', 3: 'At', 4: 'B', 5: 'Before', 6: 'Black', 7: 'C', 8: 'Can', 9: 'Choose', 10: 'Coffee', 
                  11: 'Cold', 12: 'D', 13: 'Doctor', 14: 'Drink', 15: 'E', 16: 'Eight', 17: 'F', 18: 'Favorite', 19: 'Five', 20: 'Four', 
                  21: 'Friday', 22: 'G', 23: 'Goodbye', 24: 'H', 25: 'Happy', 26: 'Has_or_Have', 27: 'Hearing_Aid', 28: 'Hello', 29: 'Help', 
                  30: 'How', 31: 'I', 32: 'I_Love_You', 33: 'I_or_Me', 34: 'J', 35: 'K', 36: 'L', 37: 'Late', 38: 'Live', 39: 'Love', 40: 'M', 
                  41: 'Monday', 42: 'Month', 43: 'My', 44: 'My_Self', 45: 'N', 46: 'Name', 47: 'Near', 48: 'No', 49: 'Now_or_Today', 50: 'O', 
                  51: 'One', 52: 'P', 53: 'Professor', 54: 'Q', 55: 'R', 56: 'S', 57: 'Saturday', 58: 'Seven', 59: 'Sit', 60: 'Sorry', 61: 'Stand',
                    62: 'Sunday', 63: 'T', 64: 'Ten', 65: 'Thank_You', 66: 'Then', 67: 'This', 68: 'Three', 69: 'Ticket', 70: 'To', 71: 'Tuesday', 
                    72: 'Two', 73: 'U', 74: 'V', 75: 'W', 76: 'Warm', 77: 'Weather', 78: 'Week', 79: 'What', 80: 'When', 81: 'Where', 82: 'White', 
                    83: 'Work', 84: 'X', 85: 'Y', 86: 'Yellow', 87: 'Yes', 88: 'You', 89: 'Your'}

######################### Function: take image and output label and its probability
# define method to predict image with the random forest model
# image -> label, probablity of it 

def model_image_predict(image):
    
    # convert image to rgb 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # get landmarks
    result = hands.process(image_rgb)
    
    # define list to put all the landmarks in it 
    all_features = []
    
    # define list to put in it x,y values for each landmarks 
    current_image_landmarks = []

    # get x and y value for each landmark

    # check if there is any detection of hands or not 
    if result.multi_hand_landmarks:

        for hand_landmark in result.multi_hand_landmarks:
            for landmark in hand_landmark.landmark:
                current_image_landmarks.append(landmark.x)
                current_image_landmarks.append(landmark.y)

        # check that the number of landmarks are equal for each image
        if len(current_image_landmarks) < 84:
            current_image_landmarks = current_image_landmarks + [0]*(84-len(current_image_landmarks))

        # append the value of current_image_data in the all_data list
        all_features.append(current_image_landmarks)
        
        # convert the all_landmarks from list to 2d array
        all_features_array = np.array(all_features)
        
        
        prediction = model.predict(all_features_array)
        prediction_with_probability = model.predict_proba(all_features_array)

        return {'class':prediction[0], 'probability':prediction_with_probability[0][prediction[0]]}
    
############################## Function: Convert the video to text
def video_to_text_prediction_english(video_path):

    # read the video
    video = cv2.VideoCapture(video_path)

    # calculate the frame per second of the video -> take 1 frame per 1/2 second
    fps = round(round(video.get(cv2.CAP_PROP_FPS))/2) # 30 frmas per sec [in each second we will get 2 frames for detection process]

    # define frame_counter variable --> when it reach 15 we will take this frame [so we take a frame after 1/2 second]
    frame_counter = 0

    # get all predictions from the video in all_prediction dictionary 
    all_predictions = dict()
    prediction_id = 0

    status = True  
    while status:
        # read frames from the video
        status, frame = video.read()

        if status == True:
            # count the current frame
            frame_counter += 1
            # here is the frame that will be used for prediction
            if frame_counter % fps == 0:
                prediction = model_image_predict(frame)
                prediction_id += 1 
                all_predictions[prediction_id] = prediction
    
    
    threshold = 0.25
    all_classes = str()
    for i in all_predictions:
        # check if there is a prediction or not
        if all_predictions[i]:
            if all_predictions[i]['probability'] > threshold:
                all_classes += index_to_class[all_predictions[i]['class']] + ' '
    # make unique classes [Cancel repetition]
    previous_class = None
    final_classes = []

    for i in all_classes.rstrip().split():

        if i != previous_class:
            final_classes.append(i)
            previous_class = i

    return " ".join(final_classes)


############################ test video_to_text_prediction_english function ########################
# print(video_to_text_prediction_english(os.path.join("english_test_videos", "i_love_you.mp4")))


################################################## text to video ########################################################################

################################### define list of the used videos for mapping 
mapping_video_list = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'about', 'after', 'at', 'b', 'before', 'black', 'c', 'can', 
                      'choose', 'coffee', 'cold', 'd', 'doctor', 'drink', 'e', 'eight', 'f', 'favorite', 'five', 'four', 'friday', 'g', 'goodbye', 
                      'h', 'happy', 'has', 'have', 'hearing_aid', 'hello', 'help', 'how', 'i', 'i_love_you', 'j', 'k', 'l', 'late', 'live', 
                      'love', 'm', 'me', 'monday', 'month', 'my', 'myself', 'n', 'name', 'near', 'nine', 'no', 'now', 'o', 'one', 'p', 'professor',
                        'q', 'r', 's', 'saturday', 'seven', 'sit', 'six', 'sorry', 'stand', 'sunday', 't', 'ten', 'thank_you', 'then', 'this', 
                        'three', 'ticket', 'to', 'today', 'tuesday', 'u', 'v', 'w', 'warm', 'weather', 'week', 'what', 'when', 'where', 'white', 
                        'work', 'x', 'y', 'yellow', 'yes', 'you', 'your', 'z', 'zero']

############################### Function to convert handle teh recived text and convert it to video 
def text_to_video_prediction_english(text:str):
    
    # convert all text to lowercase
    text = text.lower()
    
    # delete any punctuation in the text
    
    # Create a translation table for the translate function
    translator = str.maketrans('', '', string.punctuation) 
    
    # Remove punctuation 
    text = text.translate(translator)  
    
    # check the multi-words
    if 'thank you' in text:
        text = text.replace('thank you', 'thank_you')
    
    if 'hearing aid' in text:
        text = text.replace('hearing aid', 'hearing_aid')
        
    if 'i love you' in text:
        text = text.replace('i love you', 'i_love_you')
    
    print(text)
    # split the whole text into separated words
    words_list = text.split(" ")
    
    # define codecc
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # define the save path 
    output_video_path = os.path.join("created_videos", "new_created_video.avi")
    
    # define video writer object to concatenate all videos in it -> by default none may be there is no words in the mapping video list
    video_writer = None
    
    for word in words_list:
        if word in mapping_video_list:
            # means there is a video for this words 
            # access this video path 
            video_path = os.path.join("mapping_videos_english", f"{word}.mp4")
            
            # read the video
            video = cv2.VideoCapture(video_path)
            
            # extract important video features
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            if video_writer is None:
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            status = True
            while status:
                
                # read Frames
                status, frame = video.read()
                
                if status == True:
                    # insert the frames of the video in the new created video [frame by frame]
                    video_writer.write(frame)
                
            
            # we get out of the first video and we will cloase it safely now
            video.release()
        
        # has no corresponding words but we can express it by letters
        else:
            
            # split the word to its letter and express the letters by their videos
            word = " ".join(word)
            
            for letter in word:
                video_path = os.path.join("mapping_videos_english", f"{letter}.mp4")
            
                # read the video
                video = cv2.VideoCapture(video_path)

                # extract important video features
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = video.get(cv2.CAP_PROP_FPS)

                if video_writer is None:
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                status = True
                while status:

                    # read Frames
                    status, frame = video.read()

                    if status == True:
                        # insert the frames of the video in the new created video [frame by frame]
                        video_writer.write(frame)

                # we get out of the first video and we will cloase it safely now
                video.release()
            
            
    # here we insert all the frames of all videos in the new created video -> we will cloase this video safely 
    if video_writer != None:
        video_writer.release()
        return output_video_path
    
################################################### test text to video function 
# print(text_to_video_prediction_english('My Name Is Mohamed'))


    



