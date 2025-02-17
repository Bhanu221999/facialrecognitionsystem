#######################################################
import uuid ## random id generator
from streamlit_option_menu import option_menu
from settings import *
#######################################################


## Disable Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)
################################################### Defining Static Data ###############################################

st.sidebar.markdown("""
    <h2 style='color: yellow; font-size: 25px;'>Facial Recognition System</h2>
    """, unsafe_allow_html=True)

# st.sidebar.markdown("&nbsp;")  # Add some space above

# Add an image below the text
st.sidebar.image("face2.jpg", width=200)

st.sidebar.markdown("""
    <h2 style='color: #14bdad; font-size: 15px;'>Facial recognition system for access control in high-security buildings</h2>
    """, unsafe_allow_html=True)

options = ["Visitor Validation", "Visitor History", "Add to Database"]
selected_menu = option_menu(
    menu_title="Select an option:",
    options=options,
    icons=["person-fill", "clock-history", "person-plus-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color : yellow": "#fafafa"},
        "icon": {"color": "#14bdad", "font-size": "23px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#ffffffb3",
        },
        "nav-link-selected": {"background-color": "black"},
    },
)


user_color      = '#0001'
title_webapp    = "Facial Recognition System"

# Apply inline CSS for space at the top
st.sidebar.markdown("<div style='margin-top: 90px;'></div>", unsafe_allow_html=True)

# HTML/CSS to style the  FORMAT button
st.sidebar.markdown("""
    <style>
        .stButton>button {
            background-color: #14bdad; /* Change button color */
            color: white; /* Change text color */
            font-size: 16px; /* Change font size */
            padding: 8px 40px; /* Change padding */
        }
    </style>
""", unsafe_allow_html=True)

###################### Defining Static Paths ###################4
# if st.sidebar.button('Click to Clear out all data'):
if st.sidebar.button('Format data' ):
    ## Clearing Visitor Database
    shutil.rmtree(VISITOR_DB, ignore_errors=True)
    os.mkdir(VISITOR_DB)
    ## Clearing Visitor History
    shutil.rmtree(VISITOR_HISTORY, ignore_errors=True)
    os.mkdir(VISITOR_HISTORY)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
# st.write(VISITOR_HISTORY)
########################################################################################################################
def main():
    
    # selected_menu = st.sidebar.radio(
    #     "Select an option:",
    #     ['Visitor Validation', 'View Visitor History', 'Add to Database']
    # )

    if selected_menu == 'Visitor Validation':
        ## Generates a Random ID for image storage
        visitor_id = uuid.uuid1()

        ## Reading Camera Image
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()

            # convert image from opened file to np.array
            image_array         = cv2.imdecode(np.frombuffer(bytes_data,
                                                             np.uint8),
                                               cv2.IMREAD_COLOR)
            image_array_copy    = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # st.image(cv2_img)

            ## Saving Visitor History
            with open(os.path.join(VISITOR_HISTORY,
                                   f'{visitor_id}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
                st.success('Image Saved Successfully!')

                ## Validating Image
                # Detect faces in the loaded image
                max_faces   = 0
                rois        = []  # region of interests (arrays of face areas)

                ## To get location of Face from Image
                face_locations  = face_recognition.face_locations(image_array)
                ## To encode Image to numeric format
                encodesCurFrame = face_recognition.face_encodings(image_array,
                                                                  face_locations)

                ## Generating Rectangle Red box over the Image
                for idx, (top, right, bottom, left) in enumerate(face_locations):
                    # Save face's Region of Interest
                    rois.append(image_array[top:bottom, left:right].copy())

                    # Draw a box around the face and label it
                    cv2.rectangle(image_array, (left, top), (right, bottom), (0, 255, 0), 2)  # Change color to green
                    cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), (0, 255, 0), cv2.FILLED)  # Change color to green
                    # cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
                    # cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    # cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)
                    cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)


                ## Showing Image
                st.image(BGR_to_RGB(image_array), width=720)

                ## Number of Faces identified
                max_faces = len(face_locations)

                if max_faces > 0:
                    col1, col2 = st.columns(2)

                    # select selected faces in the picture
                    face_idxs = col1.multiselect("Select face#", range(max_faces),
                                                 default=range(max_faces))

                    ## Filtering for similarity beyond threshold
                    similarity_threshold = col2.slider('Select Threshold for Similarity',
                                                         min_value=0.0, max_value=1.0,
                                                         value=0.9)
                                                    ## check for similarity confidence greater than this threshold

                    flag_show = False

                    if ((col1.checkbox('Click to proceed!')) & (len(face_idxs)>0)):
                        dataframe_new = pd.DataFrame()

                        ## Iterating faces one by one
                        for face_idx in face_idxs:
                            ## Getting Region of Interest for that Face
                            roi = rois[face_idx]
                            # st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

                            # initial database for known faces
                            database_data = initialize_data()
                            # st.write(DB)

                            ## Getting Available information from Database
                            face_encodings  = database_data[COLS_ENCODE].values
                            dataframe       = database_data[COLS_INFO]

                            # Comparing ROI to the faces available in database and finding distances and similarities
                            faces = face_recognition.face_encodings(roi)
                            # st.write(faces)

                            if len(faces) < 1:
                                ## Face could not be processed
                                st.error(f'Please Try Again for face#{face_idx}!')
                            else:
                                face_to_compare = faces[0]
                                ## Comparing Face with available information from database
                                dataframe['distance'] = face_recognition.face_distance(face_encodings,
                                                                                       face_to_compare)
                                dataframe['distance'] = dataframe['distance'].astype(float)

                                dataframe['similarity'] = dataframe.distance.apply(
                                    lambda distance: f"{face_distance_to_conf(distance):0.2}")
                                dataframe['similarity'] = dataframe['similarity'].astype(float)

                                dataframe_new = dataframe.drop_duplicates(keep='first')
                                dataframe_new.reset_index(drop=True, inplace=True)
                                dataframe_new.sort_values(by="similarity", ascending=True)

                                dataframe_new = dataframe_new[dataframe_new['similarity'] > similarity_threshold].head(1)
                                dataframe_new.reset_index(drop=True, inplace=True)

                                if dataframe_new.shape[0]>0:
                                    (top, right, bottom, left) = (face_locations[face_idx])

                                    ## Save Face Region of Interest information to the list
                                    rois.append(image_array_copy[top:bottom, left:right].copy())

                                    # Draw a Rectangle Red box around the face and label it
                                    # cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
                                    # cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
                                    cv2.rectangle(image_array_copy, (left, top), (right, bottom), (0, 255, 0), 2)
                                    cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), (0, 255, 0), cv2.FILLED)

                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, 0.55, (0, 0, 0), 1)


                                    ## Getting Name of Visitor
                                    name_visitor = dataframe_new.loc[0, 'Name']
                                    attendance(visitor_id, name_visitor)

                                    flag_show = True

                                else:
                                    st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
                                    st.info('Please Update the database for a new person or click again!')
                                    attendance(visitor_id, 'Unknown')

                        if flag_show == True:
                            st.image(BGR_to_RGB(image_array_copy), width=720)

                else:
                    st.error('No human face detected.')

    if selected_menu == 'Visitor History':
        view_attendance()

    if selected_menu == 'Add to Database':
        name = st.text_input('Name:', '')
    
        upload_option = st.radio('Upload Picture', options=["Upload a Picture", "Click a picture"], horizontal=False)
    
        if upload_option == 'Upload a Picture':
            img_file_buffer = st.file_uploader('', type=allowed_image_type)
        elif upload_option == 'Click a picture':
            img_file_buffer = st.camera_input("")
    
        if ((img_file_buffer is not None) & (len(name) > 1) & st.button('Click to Save!')):
            # To read image file buffer with OpenCV
            if upload_option == 'Upload a Picture':
                file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
            elif upload_option == 'Click a picture':
                file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)

        
        # Convert image from opened file to np.array
            image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Save the image
            with open(os.path.join(VISITOR_DB, f'{name}.jpg'), 'wb') as file:
                file.write(img_file_buffer.getbuffer())
        
            # Perform face recognition
            face_locations = face_recognition.face_locations(image_array)
            encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)
            df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
            df_new[COLS_INFO] = name
            df_new = df_new[COLS_INFO + COLS_ENCODE].copy()
            
            # Initialize and update the database
            DB = initialize_data()
            DB = pd.concat([DB, df_new], ignore_index=True)
            DB.drop_duplicates(keep='first', inplace=True)
            DB.reset_index(drop=True, inplace=True)
            add_data_db(DB)

    # st.sidebar.header("About")
    # st.sidebar.info("This webapp gives a demo of Visitor Monitoring "
    # "Webapp using 'Face Recognition' and Streamlit")

#######################################################
if __name__ == "__main__":
    main()
#######################################################













# import uuid
# import os
# import shutil
# import cv2
# import numpy as np
# import face_recognition
# import pandas as pd
# import streamlit as st
# from PIL import Image

# # Constants
# COLOR_DARK = (0, 0, 139)
# COLOR_WHITE = (255, 255, 255)
# COLS_ENCODE = ['encode']
# COLS_INFO = ['name', 'age', 'gender']

# # Paths
# VISITOR_DB = "./visitor_db"
# VISITOR_HISTORY = "./visitor_history"

# def face_distance_to_conf(face_distance, face_match_threshold=0.6):
#     if face_distance > face_match_threshold:
#         range = (1.0 - face_match_threshold)
#         linear_val = (1.0 - face_distance) / (range * 2.0)
#         return linear_val
#     else:
#         range = face_match_threshold
#         linear_val = 1.0 - (face_distance / (range * 2.0))
#         return linear_val + ((1.0 - linear_val) * ((range * 2.0) - 1.0))

# def initialize_data():
#     database_data = pd.DataFrame()
#     return database_data

# def BGR_to_RGB(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# def main():
#     st.sidebar.header("About")
#     st.sidebar.info("This webapp gives a demo of Visitor Monitoring Webapp using 'Face Recognition' and Streamlit")

#     selected_menu = st.sidebar.selectbox(
#         "Select an option:",
#         ['Visitor Validation', 'View Visitor History', 'Add to Database']
#     )

#     if selected_menu == 'Visitor Validation':
#         visitor_id = uuid.uuid1()
#         img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#         if img_file_buffer is not None:
#             bytes_data = img_file_buffer.getvalue()
#             image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
#             image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

#             with open(os.path.join(VISITOR_HISTORY, f'{visitor_id}.jpg'), 'wb') as file:
#                 file.write(img_file_buffer.getbuffer())
#                 st.success('Image saved successfully!')

#                 max_faces = 0
#                 rois = []
#                 face_locations = face_recognition.face_locations(image_array)
#                 encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)

#                 for idx, (top, right, bottom, left) in enumerate(face_locations):
#                     rois.append(image_array[top:bottom, left:right].copy())
#                     cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
#                     cv2.rectangle(image_array, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
#                     font = cv2.FONT_HERSHEY_DUPLEX
#                     cv2.putText(image_array, f"#{idx}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)

#                 st.image(BGR_to_RGB(image_array), width=720)
#                 max_faces = len(face_locations)

#                 if max_faces > 0:
#                     col1, col2 = st.columns(2)
#                     face_idxs = col1.multiselect("Select face#", range(max_faces), default=range(max_faces))
#                     similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0, max_value=1.0, value=0.5)

#                     if col1.checkbox('Click to proceed!') and len(face_idxs) > 0:
#                         dataframe_new = pd.DataFrame()

#                         for face_idx in face_idxs:
#                             roi = rois[face_idx]
#                             database_data = initialize_data()
#                             face_encodings = database_data[COLS_ENCODE].values
#                             dataframe = database_data[COLS_INFO]

#                             faces = face_recognition.face_encodings(roi)
#                             if len(faces) < 1:
#                                 st.error(f'Please Try Again for face#{face_idx}!')
#                             else:
#                                 face_to_compare = faces[0]
#                                 dataframe['distance'] = face_recognition.face_distance(face_encodings, face_to_compare)
#                                 dataframe['distance'] = dataframe['distance'].astype(float)
#                                 dataframe['similarity'] = dataframe.distance.apply(lambda distance: f"{face_distance_to_conf(distance):0.2}")
#                                 dataframe['similarity'] = dataframe['similarity'].astype(float)
#                                 dataframe_new = dataframe.drop_duplicates(keep='first')
#                                 dataframe_new.reset_index(drop=True, inplace=True)
#                                 dataframe_new.sort_values(by="similarity", ascending=True)
#                                 dataframe_new = dataframe_new[dataframe_new['similarity'] > similarity_threshold].head(1)
#                                 dataframe_new.reset_index(drop=True, inplace=True)

#                                 if len(dataframe_new) > 0:
#                                     st.success("Match found!")
#                                     st.write(dataframe_new)
#                                 else:
#                                     st.warning("No match found!")

#                         col1.text('Upload an image again or select from the sidebar to view the history!')

#     elif selected_menu == 'View Visitor History':
#         image_files = os.listdir(VISITOR_HISTORY)
#         if len(image_files) > 0:
#             selected_image = st.sidebar.selectbox(
#                 "Select an image to view:",
#                 image_files
#             )
#             image_path = os.path.join(VISITOR_HISTORY, selected_image)
#             st.image(image_path, width=720)
#         else:
#             st.info("No visitor history available.")

#     elif selected_menu == 'Add to Database':
#         st.info("Option to add visitor to the database will be available soon!")

# if __name__ == "__main__":
#     main()







# COMBINED CODE SETTINGS.PY AND APP.PY WITHOUT MEDIAPIPE

# import os
# import pathlib
# import streamlit as st
# import datetime
# import json
# import sys
# import shutil
# import pandas as pd
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image
# import uuid
# from streamlit_option_menu import option_menu
# from settings import *

# ########################################################################################################################
# # The Root Directory of the project
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

# STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

# ## We create a downloads directory within the streamlit static asset directory and we write output files to it
# DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
# if not DOWNLOADS_PATH.is_dir():
#     DOWNLOADS_PATH.mkdir()

# LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
# if not LOG_DIR.is_dir():
#     LOG_DIR.mkdir()

# OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
# if not OUT_DIR.is_dir():
#     OUT_DIR.mkdir()

# VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
# VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")

# if not os.path.exists(VISITOR_DB):
#     os.mkdir(VISITOR_DB)

# if not os.path.exists(VISITOR_HISTORY):
#     os.mkdir(VISITOR_HISTORY)
# ########################################################################################################################
# ## Defining Parameters

# COLOR_DARK  = (0, 0, 153)
# COLOR_WHITE = (255, 255, 255)
# COLS_INFO   = ['Name']
# COLS_ENCODE = [f'v{i}' for i in range(128)]

# ## Database
# data_path       = VISITOR_DB
# file_db         = 'visitors_db.csv'         ## To store user information
# file_history    = 'visitors_history.csv'    ## To store visitor history information

# ## Image formats allowed
# allowed_image_type = ['.png', '.jpg', '.jpeg']
# ################################################### Defining Function ##############################################
# def initialize_data():
#     if os.path.exists(os.path.join(data_path, file_db)):
#         df = pd.read_csv(os.path.join(data_path, file_db))
#     else:
#         df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
#         df.to_csv(os.path.join(data_path, file_db), index=False)
#     return df

# #################################################################
# def add_data_db(df_visitor_details):
#     try:
#         df_all = pd.read_csv(os.path.join(data_path, file_db))
#         if not df_all.empty:
#             df_all = pd.concat([df_all, df_visitor_details], ignore_index=True)
#             df_all.drop_duplicates(keep='first', inplace=True)
#             df_all.reset_index(inplace=True, drop=True)
#             df_all.to_csv(os.path.join(data_path, file_db), index=False)
#             st.success('Details Added Successfully!')
#         else:
#             df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
#             st.success('Initiated Data Successfully!')
#     except Exception as e:
#         st.error(e)

# #################################################################
# # convert opencv BRG to regular RGB mode
# def BGR_to_RGB(image_in_array):
#     return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# #################################################################
# def face_distance_to_conf(face_distance, face_match_threshold=0.6):
#     if face_distance > face_match_threshold:
#         range = (1.0 - face_match_threshold)
#         linear_val = (1.0 - face_distance) / (range * 2.0)
#         return linear_val
#     else:
#         range = face_match_threshold
#         linear_val = 1.0 - (face_distance / (range * 2.0))
#         return linear_val + ((1.0 - linear_val) * np.power(
#             (linear_val - 0.5) * 2, 0.2))

# #################################################################
# def attendance(id, name):
#     f_p = os.path.join(VISITOR_HISTORY, file_history)
#     now = datetime.datetime.now()
#     dtString = now.strftime('%Y-%m-%d %H:%M:%S')
#     df_attendance_temp = pd.DataFrame(data={"id": [id], "visitor_name": [name], "Timing": [dtString]})
#     if not os.path.isfile(f_p):
#         df_attendance_temp.to_csv(f_p, index=False)
#     else:
#         df_attendance = pd.read_csv(f_p)
#         df_attendance = pd.concat([df_attendance, df_attendance_temp], ignore_index=True)
#         df_attendance.to_csv(f_p, index=False)

# #################################################################
# def view_attendance():
#     f_p = os.path.join(VISITOR_HISTORY, file_history)
#     df_attendance_temp = pd.DataFrame(columns=["id", "visitor_name", "Timing"])
#     if not os.path.isfile(f_p):
#         df_attendance_temp.to_csv(f_p, index=False)
#     else:
#         df_attendance_temp = pd.read_csv(f_p)
#     df_attendance = df_attendance_temp.sort_values(by='Timing', ascending=False)
#     df_attendance.reset_index(inplace=True, drop=True)
#     st.write(df_attendance)
#     if df_attendance.shape[0]>0:
#         id_chk  = df_attendance.loc[0, 'id']
#         id_name = df_attendance.loc[0, 'visitor_name']
#         selected_img = st.selectbox('Search Image using ID', options=['None']+list(df_attendance['id']))
#         avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
#                        if ((file.endswith(tuple(allowed_image_type))) & (file.startswith(selected_img) == True))]
#         if len(avail_files)>0:
#             selected_img_path = os.path.join(VISITOR_HISTORY, avail_files[0])
#             st.image(Image.open(selected_img_path))

# #######################################################
# ## Disable Warnings
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showfileUploaderEncoding', False)
# ################################################### Defining Static Data ###############################################
# st.sidebar.markdown("""
#                     > Made by [*Bhanu*](https://www.linkedin.com/in/bhanushankargubba/)
#                     """)

# user_color      = '#000000'
# title_webapp    = "Visitor Monitoring Webapp"

# def main():
#     selected_menu = st.sidebar.radio(
#         "Select an option:",
#         ['Visitor Validation', 'View Visitor History', 'Add to Database']
#     )

#     if selected_menu == 'Visitor Validation':
#         visitor_id = uuid.uuid1()
#         img_file_buffer = st.camera_input("Take a picture")
#         if img_file_buffer is not None:
#             bytes_data = img_file_buffer.getvalue()
#             image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
#             image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
#             with open(os.path.join(VISITOR_HISTORY, f'{visitor_id}.jpg'), 'wb') as file:
#                 file.write(img_file_buffer.getbuffer())
#                 st.success('Image Saved Successfully!')
#             face_locations = face_recognition.face_locations(image_array)
#             encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)
#             if len(face_locations) > 0:
#                 col1, col2 = st.columns(2)
#                 face_idxs = col1.multiselect("Select face#", range(len(face_locations)), default=range(len(face_locations)))
#                 similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0, max_value=1.0, value=0.9)
#                 flag_show = False
#                 if ((col1.checkbox('Click to proceed!')) & (len(face_idxs)>0)):
#                     dataframe_new = pd.DataFrame()
#                     for face_idx in face_idxs:
#                         roi = image_array_copy[face_locations[face_idx][0]:face_locations[face_idx][2], face_locations[face_idx][3]:face_locations[face_idx][1]].copy()
#                         database_data = initialize_data()
#                         face_encodings = database_data[COLS_ENCODE].values
#                         dataframe = database_data[COLS_INFO]
#                         faces = face_recognition.face_encodings(roi)
#                         if len(faces) < 1:
#                             st.error(f'Please Try Again for face#{face_idx}!')
#                         else:
#                             face_to_compare = faces[0]
#                             dataframe['distance'] = face_recognition.face_distance(face_encodings, face_to_compare)
#                             dataframe['distance'] = dataframe['distance'].astype(float)
#                             dataframe['similarity'] = dataframe.distance.apply(lambda distance: f"{face_distance_to_conf(distance):0.2}")
#                             dataframe['similarity'] = dataframe['similarity'].astype(float)
#                             dataframe_new = dataframe.drop_duplicates(keep='first')
#                             dataframe_new.reset_index(drop=True, inplace=True)
#                             dataframe_new.sort_values(by="similarity", ascending=True)
#                             dataframe_new = dataframe_new[dataframe_new['similarity'] > similarity_threshold].head(1)
#                             dataframe_new.reset_index(drop=True, inplace=True)
#                             if dataframe_new.shape[0]>0:
#                                 (top, right, bottom, left) = (face_locations[face_idx])
#                                 cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
#                                 cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
#                                 font = cv2.FONT_HERSHEY_DUPLEX
#                                 cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)
#                                 name_visitor = dataframe_new.loc[0, 'Name']
#                                 attendance(visitor_id, name_visitor)
#                                 flag_show = True
#                             else:
#                                 st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
#                                 st.info('Please Update the database for a new person or click again!')
#                                 attendance(visitor_id, 'Unknown')
#                     if flag_show == True:
#                         st.image(BGR_to_RGB(image_array_copy), width=720)
#             else:
#                 st.error('No human face detected.')
#     elif selected_menu == 'View Visitor History':
#         view_attendance()
#     elif selected_menu == 'Add to Database':
#         name = st.text_input('Name:', '')
#         upload_option = st.radio('Upload Picture', options=["Upload a Picture", "Click a picture"], horizontal=False)
#         if upload_option == 'Upload a Picture':
#             img_file_buffer = st.file_uploader('', type=allowed_image_type)
#         elif upload_option == 'Click a picture':
#             img_file_buffer = st.camera_input("")
#         if ((img_file_buffer is not None) & (len(name) > 1) & st.button('Click to Save!')):
#             if upload_option == 'Upload a Picture':
#                 file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
#             elif upload_option == 'Click a picture':
#                 file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)
#             image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             with open(os.path.join(VISITOR_DB, f'{name}.jpg'), 'wb') as file:
#                 file.write(img_file_buffer.getbuffer())
#             face_locations = face_recognition.face_locations(image_array)
#             encodesCurFrame = face_recognition.face_encodings(image_array, face_locations)
#             df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
#             df_new[COLS_INFO] = name
#             df_new = df_new[COLS_INFO + COLS_ENCODE].copy()
#             DB = initialize_data()
#             DB = pd.concat([DB, df_new], ignore_index=True)
#             DB.drop_duplicates(keep='first', inplace=True)
#             DB.reset_index(drop=True, inplace=True)
#             add_data_db(DB)

#     st.sidebar.header("About")
#     st.sidebar.info("This webapp gives a demo of Visitor Monitoring "
#     "Webapp using 'Face Recognition' and Streamlit")

# #######################################################
# if __name__ == "__main__":
#     main()
# #######################################################





















# COMBINED CODE SETTINGS.PY AND APP.PY WITH MEDIAPIPE


# import os
# import pathlib
# import streamlit as st
# import datetime
# import json
# import sys
# import shutil
# import pandas as pd
# import cv2
# import numpy as np
# from PIL import Image
# import uuid
# from streamlit_option_menu import option_menu
# from settings import *
# import mediapipe as mp

# ########################################################################################################################
# # The Root Directory of the project
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

# STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

# ## We create a downloads directory within the streamlit static asset directory and we write output files to it
# DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
# if not DOWNLOADS_PATH.is_dir():
#     DOWNLOADS_PATH.mkdir()

# LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
# if not LOG_DIR.is_dir():
#     LOG_DIR.mkdir()

# OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
# if not OUT_DIR.is_dir():
#     OUT_DIR.mkdir()

# VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
# VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")

# if not os.path.exists(VISITOR_DB):
#     os.mkdir(VISITOR_DB)

# if not os.path.exists(VISITOR_HISTORY):
#     os.mkdir(VISITOR_HISTORY)
# ########################################################################################################################
# ## Defining Parameters

# COLOR_DARK  = (0, 0, 153)
# COLOR_WHITE = (255, 255, 255)
# COLS_INFO   = ['Name']
# COLS_ENCODE = [f'v{i}' for i in range(128)]

# ## Database
# data_path       = VISITOR_DB
# file_db         = 'visitors_db.csv'         ## To store user information
# file_history    = 'visitors_history.csv'    ## To store visitor history information

# ## Image formats allowed
# allowed_image_type = ['.png', '.jpg', '.jpeg']
# ################################################### Defining Function ##############################################
# def initialize_data():
#     if os.path.exists(os.path.join(data_path, file_db)):
#         df = pd.read_csv(os.path.join(data_path, file_db))
#     else:
#         df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
#         df.to_csv(os.path.join(data_path, file_db), index=False)
#     return df

# #################################################################
# def add_data_db(df_visitor_details):
#     try:
#         df_all = pd.read_csv(os.path.join(data_path, file_db))
#         if not df_all.empty:
#             df_all = pd.concat([df_all, df_visitor_details], ignore_index=True)
#             df_all.drop_duplicates(keep='first', inplace=True)
#             df_all.reset_index(inplace=True, drop=True)
#             df_all.to_csv(os.path.join(data_path, file_db), index=False)
#             st.success('Details Added Successfully!')
#         else:
#             df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
#             st.success('Initiated Data Successfully!')
#     except Exception as e:
#         st.error(e)

# #################################################################
# # convert opencv BRG to regular RGB mode
# def BGR_to_RGB(image_in_array):
#     return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# #################################################################
# def face_distance_to_conf(face_distance, face_match_threshold=0.6):
#     if face_distance > face_match_threshold:
#         range = (1.0 - face_match_threshold)
#         linear_val = (1.0 - face_distance) / (range * 2.0)
#         return linear_val
#     else:
#         range = face_match_threshold
#         linear_val = 1.0 - (face_distance / (range * 2.0))
#         return linear_val + ((1.0 - linear_val) * np.power(
#             (linear_val - 0.5) * 2, 0.2))

# #################################################################
# def attendance(id, name):
#     f_p = os.path.join(VISITOR_HISTORY, file_history)
#     now = datetime.datetime.now()
#     dtString = now.strftime('%Y-%m-%d %H:%M:%S')
#     df_attendance_temp = pd.DataFrame(data={"id": [id], "visitor_name": [name], "Timing": [dtString]})
#     if not os.path.isfile(f_p):
#         df_attendance_temp.to_csv(f_p, index=False)
#     else:
#         df_attendance = pd.read_csv(f_p)
#         df_attendance = pd.concat([df_attendance, df_attendance_temp], ignore_index=True)
#         df_attendance.to_csv(f_p, index=False)

# #################################################################
# def view_attendance():
#     f_p = os.path.join(VISITOR_HISTORY, file_history)
#     df_attendance_temp = pd.DataFrame(columns=["id", "visitor_name", "Timing"])
#     if not os.path.isfile(f_p):
#         df_attendance_temp.to_csv(f_p, index=False)
#     else:
#         df_attendance_temp = pd.read_csv(f_p)
#     df_attendance = df_attendance_temp.sort_values(by='Timing', ascending=False)
#     df_attendance.reset_index(inplace=True, drop=True)
#     st.write(df_attendance)
#     if df_attendance.shape[0]>0:
#         id_chk  = df_attendance.loc[0, 'id']
#         id_name = df_attendance.loc[0, 'visitor_name']
#         selected_img = st.selectbox('Search Image using ID', options=['None']+list(df_attendance['id']))
#         avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
#                        if ((file.endswith(tuple(allowed_image_type))) & (file.startswith(selected_img) == True))]
#         if len(avail_files)>0:
#             selected_img_path = os.path.join(VISITOR_HISTORY, avail_files[0])
#             st.image(Image.open(selected_img_path))

# #######################################################
# ## Disable Warnings
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showfileUploaderEncoding', False)
# ################################################### Defining Static Data ###############################################
# st.sidebar.markdown("""
#                     > Made by [*Bhanu*](https://www.linkedin.com/in/bhanushankargubba/)
#                     """)

# user_color      = '#000000'
# title_webapp    = "Visitor Monitoring Webapp"

# def detect_faces(image_array):
#     mp_face_detection = mp.solutions.face_detection
#     with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
#         image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(image_rgb)
#         face_locations = []
#         if results.detections:
#             for detection in results.detections:
#                 location = detection.location_data.relative_bounding_box
#                 h, w, c = image_array.shape
#                 face_locations.append((
#                     int(location.ymin * h),
#                     int((location.xmin + location.width) * w),  # Adjusted here
#                     int((location.ymin + location.height) * h), # Adjusted here
#                     int(location.xmin * w)
#                 ))

#         return face_locations

# def extract_face_embeddings(image_array, face_locations):
#     mp_face_recognition = mp.solutions.face_recognition
#     with mp_face_recognition.FaceRecognition(min_detection_confidence=0.5) as face_recognition:
#         image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
#         results = face_recognition.process(image_rgb)
#         face_embeddings = []
#         if results.face_embeddings:
#             for embedding in results.face_embeddings:
#                 face_embeddings.append(embedding)
#         return face_embeddings

# def main():
#     selected_menu = st.sidebar.radio(
#         "Select an option:",
#         ['Visitor Validation', 'View Visitor History', 'Add to Database']
#     )

#     if selected_menu == 'Visitor Validation':
#         visitor_id = uuid.uuid1()
#         img_file_buffer = st.camera_input("Take a picture")
#         if img_file_buffer is not None:
#             bytes_data = img_file_buffer.getvalue()
#             image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
#             image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
#             with open(os.path.join(VISITOR_HISTORY, f'{visitor_id}.jpg'), 'wb') as file:
#                 file.write(img_file_buffer.getbuffer())
#                 st.success('Image Saved Successfully!')
#             face_locations = detect_faces(image_array)
#             encodesCurFrame = extract_face_embeddings(image_array, face_locations)
#             if len(face_locations) > 0:
#                 col1, col2 = st.columns(2)
#                 face_idxs = col1.multiselect("Select face#", range(len(face_locations)), default=range(len(face_locations)))
#                 similarity_threshold = col2.slider('Select Threshold for Similarity', min_value=0.0, max_value=1.0, value=0.9)
#                 flag_show = False
#                 if ((col1.checkbox('Click to proceed!')) & (len(face_idxs)>0)):
#                     dataframe_new = pd.DataFrame()
#                     for face_idx in face_idxs:
#                         roi = image_array_copy[face_locations[face_idx][0]:face_locations[face_idx][2], face_locations[face_idx][3]:face_locations[face_idx][1]].copy()
#                         database_data = initialize_data()
#                         face_encodings = database_data[COLS_ENCODE].values
#                         dataframe = database_data[COLS_INFO]
#                         faces = extract_face_embeddings(roi, [(0, roi.shape[1], roi.shape[0], 0)])
#                         if len(faces) < 1:
#                             st.error(f'Please Try Again for face#{face_idx}!')
#                         else:
#                             face_to_compare = faces[0]
#                             dataframe['distance'] = [np.linalg.norm(face_encoding - face_to_compare) for face_encoding in face_encodings]
#                             dataframe['distance'] = dataframe['distance'].astype(float)
#                             dataframe['similarity'] = dataframe.distance.apply(lambda distance: f"{face_distance_to_conf(distance):0.2}")
#                             dataframe['similarity'] = dataframe['similarity'].astype(float)
#                             dataframe_new = dataframe.drop_duplicates(keep='first')
#                             dataframe_new.reset_index(drop=True, inplace=True)
#                             dataframe_new.sort_values(by="similarity", ascending=True)
#                             dataframe_new = dataframe_new[dataframe_new['similarity'] > similarity_threshold].head(1)
#                             dataframe_new.reset_index(drop=True, inplace=True)
#                             if dataframe_new.shape[0]>0:
#                                 (top, right, bottom, left) = (face_locations[face_idx])
#                                 cv2.rectangle(image_array_copy, (left, top), (right, bottom), COLOR_DARK, 2)
#                                 cv2.rectangle(image_array_copy, (left, bottom + 35), (right, bottom), COLOR_DARK, cv2.FILLED)
#                                 font = cv2.FONT_HERSHEY_DUPLEX
#                                 cv2.putText(image_array_copy, f"#{dataframe_new.loc[0, 'Name']}", (left + 5, bottom + 25), font, .55, COLOR_WHITE, 1)
#                                 name_visitor = dataframe_new.loc[0, 'Name']
#                                 attendance(visitor_id, name_visitor)
#                                 flag_show = True
#                             else:
#                                 st.error(f'No Match Found for the given Similarity Threshold! for face#{face_idx}')
#                                 st.info('Please Update the database for a new person or click again!')
#                                 attendance(visitor_id, 'Unknown')
#                     if flag_show == True:
#                         st.image(BGR_to_RGB(image_array_copy), width=720)
#             else:
#                 st.error('No human face detected.')
#     elif selected_menu == 'View Visitor History':
#         view_attendance()
#     elif selected_menu == 'Add to Database':
#         name = st.text_input('Name:', '')
#         upload_option = st.radio('Upload Picture', options=["Upload a Picture", "Click a picture"], horizontal=False)
#         if upload_option == 'Upload a Picture':
#             img_file_buffer = st.file_uploader('', type=allowed_image_type)
#         elif upload_option == 'Click a picture':
#             img_file_buffer = st.camera_input("")
#         if ((img_file_buffer is not None) & (len(name) > 1) & st.button('Click to Save!')):
#             if upload_option == 'Upload a Picture':
#                 file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
#             elif upload_option == 'Click a picture':
#                 file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)
#             image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             with open(os.path.join(VISITOR_DB, f'{name}.jpg'), 'wb') as file:
#                 file.write(img_file_buffer.getbuffer())
#             face_locations = detect_faces(image_array)
#             encodesCurFrame = extract_face_embeddings(image_array, face_locations)
#             df_new = pd.DataFrame(data=encodesCurFrame, columns=COLS_ENCODE)
#             df_new[COLS_INFO] = name
#             df_new = df_new[COLS_INFO + COLS_ENCODE].copy()
#             DB = initialize_data()
#             DB = pd.concat([DB, df_new], ignore_index=True)
#             DB.drop_duplicates(keep='first', inplace=True)
#             DB.reset_index(drop=True, inplace=True)
#             add_data_db(DB)

#     st.sidebar.header("About")
#     st.sidebar.info("This webapp gives a demo of Visitor Monitoring "
#     "Webapp using 'Face Recognition' and Streamlit")

# #######################################################
# if __name__ == "__main__":
#     main()
# #######################################################