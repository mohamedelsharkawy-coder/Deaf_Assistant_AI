##################### libraries
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import bz2file as bz2

########################################  video to text  ####################################################################################

##################### load google model 
# create object just focus on the hands 
mp_hands = mp.solutions.hands

# get the model that detect hand_landmarks 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

###################### load my model --> the original model file with no compression
# model_file = open('arabic_model.pickle','rb')
# model_dict = pickle.load(model_file)
# model = model_dict['model']
# model_file.close()

########################### define a function that decompressed the compressed file 
########## load my model --> the compressed model
# def decompress_pickle(file):
#     data = bz2.BZ2File(file, 'rb')
#     data = pickle.load(data)
#     return data

# model = decompress_pickle("compressed_arabic_model.pbz2")

##########################################
# def decompress_pickle(file):
#     with bz2.BZ2File(file, 'rb') as f:
#         data = pickle.load(f)
#     return data

# model = decompress_pickle("compressed_arabic_model.pbz2")

#######################################
# # load the model as the compressed version 18 mb, not the whole 600mb
# def load_compressed_model(file_path):
#     with bz2.open(file_path, 'rb') as f:
#         return pickle.load(f)

# model = load_compressed_model('compressed_arabic_model.pbz2')

######################## label map dictionary
index_to_class_english = {0: 'aeen', 1: 'alef', 2: 'allah', 3: 'bad', 4: 'bank', 5: 'bathroom', 6: 'beh', 7: 'cairo', 8: 'chair', 9: 'child', 
                  10: 'college', 11: 'daaad', 12: 'daal', 13: 'deaf', 14: 'dish', 15: 'divorce', 16: 'egypt', 17: 'eight', 18: 'engagement', 
                  19: 'engineer', 20: 'father', 21: 'feh', 22: 'five', 23: 'four', 24: 'geem', 25: 'gheen', 26: 'girl_or_daughter', 27: 'green', 
                  28: 'haa', 29: 'happy', 30: 'hehh', 31: 'help', 32: 'home_or_house', 33: 'hospital', 34: 'hungry', 35: 'ismaalia', 
                  36: 'i_love_you', 37: 'i_or_me', 38: 'kaaf', 39: 'khaa', 40: 'laam', 41: 'luxor', 42: 'meem', 43: 'monday', 44: 'nine', 
                  45: 'noon', 46: 'one', 47: 'orange', 48: 'pray', 49: 'qaaf', 50: 'raa', 51: 'saaad', 52: 'saturday', 53: 'school', 54: 'seen', 
                  55: 'seven', 56: 'sharkia', 57: 'sheen', 58: 'sign', 59: 'six', 60: 'sleep_n_v', 61: 'son_or_boy', 62: 'tah', 63: 'teacher', 
                  64: 'teen_fruit', 65: 'teh', 66: 'ten', 67: 'theh', 68: 'the_peace', 69: 'three', 70: 'thursday', 71: 'tuesday', 72: 'tv', 
                  73: 'two', 74: 'university', 75: 'waow', 76: 'water', 77: 'wednesday', 78: 'what', 79: 'where', 80: 'white', 81: 'work_v', 
                  82: 'yeeh', 83: 'yellow', 84: 'yes', 85: 'you', 86: 'your_name', 87: 'zaal', 88: 'zah', 89: 'zeen'}

index_to_class_english_to_arabic = {'aeen':'ع', 'alef':'أ', 'allah':'الله', 'bad':'سىء', 'bank':'بنك', 'bathroom':'حمام',
                        'beh':'ب', 'cairo':'القاهرة', 'chair':'كرسي', 'child':'طفل', 'college':'كلية', 'daaad':'ض',
                        'daal':'د', 'deaf':'أصم', 'dish':'طبق', 'divorce':'طلاق/مطلق/مطلقة', 'egypt':'مصر', 'eight':'ثمانية',
                        'engagement':'خطوبة', 'engineer':'مهندس', 'father':'أب', 'feh':'ف', 'five':'خمسة', 'four':'أربعة',
                        'geem':'ج', 'gheen':'غ', 'girl_or_daughter':'بنت/أبنة', 'green':'أخضر', 'haa':'ح', 'happy':'سعيد',
                        'hehh':'ه', 'help':'مساعدة', 'home_or_house':'منزل/بيت', 'hospital':'مستشفى', 'hungry':'جائع', 'ismaalia':'الاسماعيلية',
                        'i_love_you':'أحبك', 'i_or_me':'انا', 'kaaf':'ك', 'khaa':'خ', 'laam':'ل', 'luxor':'الأقصر',
                        'meem':'م', 'monday':'الأثنين', 'nine':'تسعة', 'noon':'ن', 'one':'واحد', 'orange':'برتقال',
                        'pray':'الصلاة', 'qaaf':'ق', 'raa':'ر', 'saaad':'ص', 'saturday':'السبت', 'school':'مدرسة',
                        'seen':'س', 'seven':'سبعة', 'sharkia':'الشرقية', 'sheen':'ش', 'sign':'اشارة', 'six':'ستة',
                        'sleep_n_v':'ينام/نوم', 'son_or_boy':'أبن/ولد', 'tah':'ط', 'teacher':'أستاذ/معلم', 'teen_fruit':'تين', 'teh':'ت',
                        'ten':'عشرة', 'theh':'ث', 'the_peace':'السلام', 'three':'ثلاثة', 'thursday':'الخميس', 'tuesday':'الثلاثاء',
                        'tv':'التيليفزيون', 'two':'أثنين', 'university':'الجامعة', 'waow':'و', 'water':'ماء', 'wednesday':'الأربعاء',
                        'what':'أيه/ماذا', 'where':'فين/أين', 'white':'أبيض', 'work_v':'يعمل/يشتغل', 'yeeh':'ي', 'yellow':'أصفر',
                        'yes':'نعم', 'you':'أنت', 'your_name':'أسمك', 'zaal':'ذ', 'zah':'ظ', 'zeen':'ز'}

########################################## Function: take image and output label and its probability

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


############################################ Function: Convert the video to text
def video_to_text_prediction_arabic(video_path):

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
                arabic_text = index_to_class_english_to_arabic[index_to_class_english[all_predictions[i]['class']]]
                all_classes += arabic_text + ' '
    # make unique classes [Cancel repetition]
    previous_class = None
    final_classes = []

    for i in all_classes.rstrip().split():

        if i != previous_class:
            final_classes.append(i)
            previous_class = i

    return " ".join(final_classes)

############################################################ test video to text funciton 
# print(video_to_text_prediction_arabic(os.path.join("one_two_three.mp4")))



########################################  text to viddeo  ####################################################################################

################################### define list of the used videos for mapping 
mapping_video_list = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'ء', 'آ', 'أ', 'أب', 'أبن', 'أبيض', 'أثنبن', 'أحب', 'أحبك', 
                      'أحمر', 'أخضر', 'أخوات', 'أربعة', 'أربعه', 'أرمل', 'أرملة', 'أرمله', 'أزرق', 'أسبوع', 'أسمك', 'أسمي', 'أشارة', 
                      'أشاره', 'أصفر', 'أصم', 'أم', 'أنا', 'أنت', 'أين', 'أيه', 'ؤ', 'إ', 'إشارة', 'إشاره', 'ا', 'اب', 'ابن', 'ابيض', 
                      'اثنين', 'احب', 'احبك', 'احمر', 'اخضر', 'اخوات', 'اربعة', 'اربعه', 'ارمل', 'ارملة', 'ارمله', 'ازرق', 'اسبوع', 
                      'اسمك', 'اسمي', 'اشارة', 'اشاره', 'اصفر', 'اصم', 'الأبيض', 'الأثنين', 'الأحد', 'الأحمر', 'الأخضر', 'الأخوات', 'الأربعاء', 
                      'الأزرق', 'الأسماعيلية', 'الأسماعيليه', 'الأصفر', 'الأقصر', 'الإسماعيلية', 'الإسماعيليه', 'الابيض', 'الاثنين', 'الاحد', 
                      'الاحمر', 'الاخضر', 'الاخوات', 'الاربعاء', 'الازرق', 'الاسماعيلية', 'الاسماعيليه', 'الاصفر', 'الاقصر', 'البحر', 'البحرالأحمر', 
                      'البحرالاحمر', 'البيت', 'الترابيزة', 'الترابيزه', 'التعارف', 'التعليم', 'التيليفيزيون', 'التين', 'الثلاثاء', 'الجمعة', 
                      'الجمعه', 'الحمام', 'الحمد', 'الحمدلله', 'الخميس', 'الرحمة', 'الرحمه', 'السبت', 'السلام', 'الشرقية', 'الشرقيه', 
                      'الشمال', 'الشهر', 'الصلاة', 'الصلاه', 'الطاولة', 'الطاوله', 'الطبق', 'الطفل', 'الطلاق', 'الظهر', 'العشاء', 'العصر', 
                      'العمر', 'الفجر', 'القاهرة', 'القاهره', 'الكلية', 'الكليه', 'الله', 'الماء', 'المدرس', 'المدرسة', 'المدرسه', 
                      'المساعدة', 'المساعده', 'المستشفى', 'المعلم', 'المغرب', 'المنزل', 'المنصورة', 'المنصوره', 'المهندس', 'الموز', 
                      'المياة', 'النوم', 'الولد', 'اليسار', 'اليمين', 'ام', 'انا', 'انت', 'اين', 'ايه', 'ب', 'بتشتغل', 'بتعمل', 'بحب', 
                      'بحر', 'برتقال', 'بنت', 'بنك', 'بني', 'بيت', 'ة', 'ت', 'تدرس', 'ترابيزة', 'ترابيزه', 'تسعة', 'تسعه', 'تسكن', 'تعارف', 
                      'تعبان', 'تعليم', 'تعمل', 'تيليفيزيون', 'تين', 'ث', 'ثلاثة', 'ثلاثه', 'ثمانية', 'ثمانيه', 'ج', 'جائع', 'جامعة', 'جامعه', 
                      'جد', 'جدة', 'جعان', 'جوعان', 'جيد', 'ح', 'حزين', 'حمام', 'حمد', 'خ', 'خطوبة', 'خطوبه', 'خمسة', 'خمسه', 'د', 'ذ', 'ر', 
                      'رحمة', 'رحمه', 'ز', 'س', 'سئ', 'ساكن', 'سبعة', 'سبعه', 'ستة', 'سته', 'سرير', 'سعيد', 'سلام', 'سنة', 'سنه', 'سىء', 
                      'سيء', 'ش', 'شاي', 'شبعان', 'شمال', 'شهر', 'ص', 'صفر', 'ض', 'ط', 'طاولة', 'طاوله', 'طبق', 'طفل', 'طلاق', 'ظ', 'ع', 
                      'عشرة', 'عشره', 'عليكم', 'عندك', 'غ', 'ف', 'فين', 'ق', 'قهوة', 'قهوه', 'ك', 'كرسي', 'كلية', 'كليه', 'كوب', 'كوباية', 
                      'كوبايه', 'ل', 'لا', 'لله', 'م', 'ماء', 'ماذا', 'متزوج', 'محتاج', 'مدرس', 'مدرسة', 'مدرسه', 'مريض', 'مساعدة', 'مساعده', 
                      'مستشفى', 'مصر', 'مطلق', 'مطلقة', 'مطلقه', 'معلم', 'منزل', 'مهندس', 'موز', 'مياة', 'مياه', 'ن', 'نتعرف', 'نعم', 'نوم', 
                      'ه', 'و', 'واحد', 'ورحمة', 'ورحمه', 'وعليكم', 'ولد', 'ى', 'ي', 'يأكل', 'ياكل', 'يحب', 'يدرس', 'يسار', 'يسكن', 'يشرب', 
                      'يعمل', 'يغلق', 'يفتح', 'يمين', 'ينام', '٠', '١', '١٠', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩']

############################### Function to convert handle teh recived text and convert it to video 
def text_to_video_prediction_arabic(text:str):
    
    # convert all text to lowercase
    text = text.lower()
    
    # delete any punctuation in the text
    
    # Create a translation table for the translate function
    translator = str.maketrans('', '', '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}%~؟%!#$"،,.ٍِـ،/:"><؛×÷‘ًًٌَُ$#@!%^&*)(''') 
    
    # Remove punctuation 
    text = text.translate(translator)  
    
    # print(text)
    # split the whole text into separated words
    words_list = text.split(" ")
    
    # define codecc
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # define the save path 
    output_video_path = os.path.join("created_videos", f"new_created_video.avi")
    
    # define video writer object to concatenate all videos in it -> by default none may be there is no words in the mapping video list
    video_writer = None
    
    for word in words_list:
        if word in mapping_video_list:
            # means there is a video for this words 
            # access this video path 
            video_path = os.path.join("mapping_videos_arabic", f"{word}.mp4")
            
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
                if letter == "ة":
                    video_path = os.path.join("mapping_videos_arabic", "ه.mp4")
                else:
                    video_path = os.path.join("mapping_videos_arabic", f"{letter}.mp4")
            
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
# print(text_to_video_prediction_arabic("انا اسمي محمد و عندي 22 سنة"))
