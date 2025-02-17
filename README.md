# Facial Recognition System for Access Control

## Overview

This project is a **Facial Recognition System** designed for access control in high-security buildings such as government offices, security infrastructures, and other sensitive areas. The system uses advanced computer vision techniques and machine learning algorithms to accurately identify individuals and control access to secure facilities. The system is built using Python, Streamlit, and the `face_recognition` library, which leverages deep learning models like ResNet and CNN for facial recognition.

The system provides a user-friendly web interface for administrators to manage visitor access, view visitor history, and add new visitors to the database. It also ensures high accuracy in identifying faces under various conditions such as different lighting, angles, and expressions.

## Features

- **Visitor Validation**: Captures live images of visitors through a webcam and validates their identity against the database.
- **Visitor History**: Maintains a log of visitor details, including name, ID, date, and time of visit.
- **Add to Database**: Allows administrators to add new visitors by uploading images or capturing them via a webcam.
- **Real-Time Processing**: Processes images in real-time with high accuracy and minimal false positives/negatives.
- **User-Friendly Interface**: Built using Streamlit, providing an interactive and responsive UI.
- **Security Protocols**: Ensures visitor data is confidential and only accessible to authorized personnel.

## Technologies Used

- **Python**: Primary programming language for the project.
- **Streamlit**: Used for building the web application interface.
- **face_recognition**: A Python library that uses deep learning models (ResNet, CNN) for facial recognition.
- **OpenCV**: Used for image processing and face detection.
- **Pandas**: For managing and storing visitor data in CSV format.
- **NumPy**: For numerical computations and image array manipulations.

## System Architecture

The system follows a **Software Development Life Cycle (SDLC)** approach, with the following phases:

1. **Requirement Gathering**: Identified functional and non-functional requirements.
2. **Design**: Designed the system architecture, including data acquisition, model training, and integration.
3. **Implementation**: Developed the system using Python and Streamlit, integrating facial recognition algorithms.
4. **Testing**: Tested the system under various conditions to ensure accuracy and performance.
5. **Deployment**: Deployed the system as a web application using Streamlit.
6. **Maintenance**: Ongoing monitoring and updates to improve system performance.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/facial-recognition-system.git
   cd facial-recognition-system
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the web application**:
   Open your browser and navigate to `http://localhost:8501`.

## Usage

1. **Visitor Validation**:
   - Capture a live image of the visitor using the webcam.
   - The system will validate the visitor's identity against the database and display the result.

2. **Visitor History**:
   - View the history of all visitors, including their name, ID, date, and time of visit.
   - Download the attendance history as a CSV file.

3. **Add to Database**:
   - Add a new visitor by uploading an image or capturing one via the webcam.
   - Enter the visitor's name and save the details to the database.

## Testing

The system has been tested under various conditions to ensure accuracy and performance. Below are some of the test cases:

| Test Case | Test Data | Expected Result |
|-----------|-----------|-----------------|
| Adding a visitor to the database | Entering name and uploading an image | Image will be updated in the database and shows a successful message. |
| Validating the visitor | Capturing the image through live camera | Image should be identified and show the name of the person. |
| Visitor history should be updated after successful detection | Shows id, name, date and time | Person data will be updated in the database table. |
| Searching the visitor with his id | Selecting the id | The visitor will be visible after giving valid id. |
| Identifying multiple people at a time | Capturing the images through live camera | Two or more people should be detected with their name at a time. |

## User Interface

<img width="590" alt="Screenshot 2025-02-17 at 11 41 56 AM" src="https://github.com/user-attachments/assets/bc122d4a-195c-4f6c-93f8-0fddc724a59b" />

<img width="433" alt="Screenshot 2025-02-17 at 11 43 22 AM" src="https://github.com/user-attachments/assets/a99be8e6-9e9a-416b-9226-2812aea68259" />


## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [face_recognition Library](https://github.com/ageitgey/face_recognition)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
