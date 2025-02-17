import base64
import os
import pathlib
import streamlit as st
import datetime
import json
import sys
import shutil
import pandas as pd
import cv2
import face_recognition
import numpy as np
from PIL import Image

########################################################################################################################
# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

## We create a downloads directory within the streamlit static asset directory and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()

OUT_DIR = (STREAMLIT_STATIC_PATH / "output")
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

VISITOR_DB = os.path.join(ROOT_DIR, "visitor_database")
# st.write(VISITOR_DB)

if not os.path.exists(VISITOR_DB):
    os.mkdir(VISITOR_DB)

VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")
# st.write(VISITOR_HISTORY)

if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
########################################################################################################################
## Defining Parameters

COLOR_DARK  = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO   = ['Name']
COLS_ENCODE = [f'v{i}' for i in range(128)]

## Database
data_path       = VISITOR_DB
file_db         = 'visitors_db.csv'         ## To store user information
file_history    = 'visitors_history.csv'    ## To store visitor history information

## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']
################################################### Defining Function ##############################################
def initialize_data():
    if os.path.exists(os.path.join(data_path, file_db)):
        # st.info('Database Found!')
        df = pd.read_csv(os.path.join(data_path, file_db))

    else:
        # st.info('Database Not Found!')
        df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
        df.to_csv(os.path.join(data_path, file_db), index=False)

    return df

#################################################################
def add_data_db(df_visitor_details):
    try:
        df_all = pd.read_csv(os.path.join(data_path, file_db))

        if not df_all.empty:
            df_all = pd.concat([df_all, df_visitor_details], ignore_index=True)
            df_all.drop_duplicates(keep='first', inplace=True)
            df_all.reset_index(inplace=True, drop=True)
            df_all.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Details Added Successfully!')
        else:
            df_visitor_details.to_csv(os.path.join(data_path, file_db), index=False)
            st.success('Initiated Data Successfully!')

    except Exception as e:
        st.error(e)

#################################################################
# convert opencv BRG to regular RGB mode
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

#################################################################
def findEncodings(images):
    encode_list = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

#################################################################
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power(
            (linear_val - 0.5) * 2, 0.2))

#################################################################
def attendance(id, name):
    f_p = os.path.join(VISITOR_HISTORY, file_history)

    now = datetime.datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    df_attendance_temp = pd.DataFrame(data={"id": [id],
                                           "visitor_name": [name],
                                           "Timing": [dtString]})

    if not os.path.isfile(f_p):
        df_attendance_temp.to_csv(f_p, index=False)
    else:
        df_attendance = pd.read_csv(f_p)
        df_attendance = pd.concat([df_attendance, df_attendance_temp], ignore_index=True)
        df_attendance.to_csv(f_p, index=False)
        # st.write(df_attendance)

#################################################################
def view_attendance():
    f_p = os.path.join(VISITOR_HISTORY, file_history)
    df_attendance_temp = pd.DataFrame(columns=["id", "visitor_name", "Timing"])

    if not os.path.isfile(f_p):
        df_attendance_temp.to_csv(f_p, index=False)
    else:
        df_attendance_temp = pd.read_csv(f_p)

    # Convert 'Timing' column to datetime format
    df_attendance_temp['Timing'] = pd.to_datetime(df_attendance_temp['Timing'])

    # Split 'Timing' column into separate 'Date' and 'Time' columns
    df_attendance_temp['Date'] = df_attendance_temp['Timing'].dt.date
    df_attendance_temp['Time'] = df_attendance_temp['Timing'].dt.time

    # Sort the DataFrame by 'Timing' column in descending order
    df_attendance = df_attendance_temp.sort_values(by='Timing', ascending=False)
    df_attendance.reset_index(drop=True, inplace=True)

    # Display the DataFrame with separate columns for date and time
    if df_attendance.shape[0] > 0:
        table = df_attendance[['id', 'visitor_name', 'Date', 'Time']]

        # Set the width of the table
        st.dataframe(table.style.set_table_styles([{'selector': 'table', 'props': [('width', '100%')]}]), width=1000)

        # Create a download button for the attendance history
        csv = df_attendance.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="attendance_history.csv">' \
               f'<button style="background-color: #14bdad; color: white; border-radius: 8px; font-size: 16px; ' \
               f'padding: 8px 40px;">Download Attendance History</button></a>'
        st.markdown(href, unsafe_allow_html=True)

        # Display image based on selected ID
        id_chk = df_attendance.loc[0, 'id']
        id_name = df_attendance.loc[0, 'visitor_name']

        selected_img = st.selectbox('Search Image using ID', options=['None'] + list(df_attendance['id']))

        avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
                       if file.endswith(tuple(allowed_image_type)) and file.startswith(selected_img)]

        if len(avail_files) > 0:
            selected_img_path = os.path.join(VISITOR_HISTORY, avail_files[0])
            st.image(Image.open(selected_img_path))





########################################################################################################################


















# import os, pathlib
# import streamlit as st
# import os, datetime, json, sys, pathlib, shutil
# import pandas as pd
# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# from PIL import Image
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
# # st.write(VISITOR_DB)

# if not os.path.exists(VISITOR_DB):
#     os.mkdir(VISITOR_DB)

# VISITOR_HISTORY = os.path.join(ROOT_DIR, "visitor_history")
# # st.write(VISITOR_HISTORY)

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
# allowed_image_type = ['.png', 'jpg', '.jpeg']
# ################################################### Defining Function ##############################################
# def initialize_data():
#     if os.path.exists(os.path.join(data_path, file_db)):
#         # st.info('Database Found!')
#         df = pd.read_csv(os.path.join(data_path, file_db))

#     else:
#         # st.info('Database Not Found!')
#         df = pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)
#         df.to_csv(os.path.join(data_path, file_db), index=False)

#     return df

# #################################################################
# def add_data_db(df_visitor_details):
#     try:
#         df_all = pd.read_csv(os.path.join(data_path, file_db))

#         if not df_all.empty:
#             df_all = df_all.append(df_visitor_details, ignore_index=False)
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
# def findEncodings(images):
#     encode_list = []

#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encode_list.append(encode)

#     return encode_list

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
#     # st.write(f_p)

#     now = datetime.datetime.now()
#     dtString = now.strftime('%Y-%m-%d %H:%M:%S')
#     df_attendace_temp = pd.DataFrame(data={ "id"            : [id],
#                                             "visitor_name"  : [name],
#                                             "Timing"        : [dtString]
#                                             })

#     if not os.path.isfile(f_p):
#         df_attendace_temp.to_csv(f_p, index=False)
#         # st.write(df_attendace_temp)
#     else:
#         df_attendace = pd.read_csv(f_p)
#         df_attendace = df_attendace.append(df_attendace_temp)
#         df_attendace.to_csv(f_p, index=False)
#         # st.write(df_attendace)

# #################################################################``
# def view_attendace():
#     f_p = os.path.join(VISITOR_HISTORY, file_history)
#     # st.write(f_p)
#     df_attendace_temp = pd.DataFrame(columns=["id",
#                                               "visitor_name", "Timing"])

#     if not os.path.isfile(f_p):
#         df_attendace_temp.to_csv(f_p, index=False)
#     else:
#         df_attendace_temp = pd.read_csv(f_p)

#     df_attendace = df_attendace_temp.sort_values(by='Timing',
#                                                  ascending=False)
#     df_attendace.reset_index(inplace=True, drop=True)

#     st.write(df_attendace)

#     if df_attendace.shape[0]>0:
#         id_chk  = df_attendace.loc[0, 'id']
#         id_name = df_attendace.loc[0, 'visitor_name']

#         selected_img = st.selectbox('Search Image using ID',
#                                     options=['None']+list(df_attendace['id']))

#         avail_files = [file for file in list(os.listdir(VISITOR_HISTORY))
#                        if ((file.endswith(tuple(allowed_image_type))) &
#                                                                               (file.startswith(selected_img) == True))]

#         if len(avail_files)>0:
#             selected_img_path = os.path.join(VISITOR_HISTORY,
#                                              avail_files[0])
#             #st.write(selected_img_path)

#             ## Displaying Image
#             st.image(Image.open(selected_img_path))


# ########################################################################################################################












