# Import necessary libraries
import streamlit as st  # frontend framework
from streamlit_lottie import st_lottie  # for lottie file rendering
import numpy as np  # for array/matrix manipulations
import tensorflow as tf  # for loading CNN EfficientNetB4 model
import cv2  # for image pre-processing
import requests  # for fetching lottie file from url
import face_recognition  # for recognizing face locations in input image
import hashlib  # for encrypting user's  password
import sqlite3  # to store user credentials
import base64   # to set dynamic/custom background image

# Set Page Configurations
st.set_page_config(
    page_title='Truth-Scan web app',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='auto'
)


@st.cache_resource()
def load_model():
    """
    To load the model for analyzing image
    Model used: EfficientNetB4
    Layers: 477
    Parameter count: 17.6 million
    Used for: image classification tasks i.e. to classify
    image as REAL/FAKE

    """
    model = tf.keras.models.load_model('.models/model.h5')
    return model


def preprocess_image(image):

    """
    To resize any input image irrespective of dimensions to
    224 x 224 to feed as the pre-processed image into model
    making it compatible for the model.
    Param: Input image
    """
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def create_users_table():
    """
    To create a table with two columns i.e.
    username & password to store user account
    credentials where username is primary key
    by establishing connection to SQLite3 DB

    """
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()


create_users_table()


def hash_password(password):
    """
    To encrypt the password via SHA-256 hashing algorithm for added security
    Param: unencrypted password
    """
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username, password):
    """
    To authenticate user while logging in to web app by checking if
    credentials entered are valid or not by matching them with users.db
    file created to store user credentials in above code.
    Params: username, password

    """
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user:
        hashed_password = user[1]
        return hashed_password == hash_password(password)
    return False


def signup(username, password):
    """
    To create new user account for those signing up for the first time.
    Params: new username, new password

    """
    hashed_password = hash_password(password)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def logout():
    """
    To end user session and logout from home page of web app.

    """
    if 'username' in st.session_state:
        del st.session_state['username']
        st.success("Successfully logged out!")
    else:
        st.warning("No user is currently logged in.")


def load_lottieUrl(url: str):
    """
    To fetch url of lottie file via requests module and return
    its json format to parse and render the animation.
    Param: Lottie url
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def authentication_page():
    """
    Frontend part of Streamlit for the authentication (Login/Sign Up) page.

    """
    def set_bg_from_url():
        """
        To unpack an image from url and set as background image

        """
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background: url("https://mir-s3-cdn-cf.behance.net/project_modules/disp/9c0722106004343.5f85fead2894a.gif");
                 background-size: 100% 100%;
                 background-repeat: no-repeat;
             }}
             </style>
             """,
            unsafe_allow_html=True
        )

    set_bg_from_url()
    if 'username' in st.session_state:
        return True

    # Center-align the title using HTML div tags with inline CSS
    st.markdown("""
        <div style="text-align: center;">
            <h1>Truth-Scan Authentication</h1>
        </div>
    """, unsafe_allow_html=True)

    lottie_hello = load_lottieUrl("https://lottie.host/a8b55a8a-79aa-42c4-a307-62ad3a874dd7/Z8w8idxTar.json")

    col1, col2, col3 = st.columns([2, 1.2, 2])
    # Display the animation in the middle column
    with col2:
        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",  # medium ; high
            height=300,
            width=300,
            key=None,
        )

    # Catchphrase for web app
    st.markdown("""
                <div style="text-align: center;">
                    <h3>Reality or Deception? We'll be your digital Sherlock!</h3>
                </div>
        """, unsafe_allow_html=True)
    buff, col, buff2 = st.columns([2, 2, 2])

    # Choose either Login/Sign Up radio button
    action = col.radio("Choose an action:", ("Login", "Sign Up"))

    # Login logic
    if action == "Login":
        buff, col, buff2 = st.columns([2, 2, 2])
        username = col.text_input("Username")
        password = col.text_input("Password", type="password")
        if col.button("Login"):
            if authenticate(username, password):
                st.session_state['username'] = username
                st.experimental_rerun()
                return True
            else:
                st.error("Invalid username or password")

    # Sign Up logic
    elif action == "Sign Up":
        buff, col, buff2 = st.columns([2, 2, 2])
        new_username = col.text_input("New Username")
        new_password = col.text_input("New Password", type="password")
        if col.button("Sign Up"):
            if signup(new_username, new_password):
                st.session_state['username'] = new_username
                st.experimental_rerun()
                st.success("Account created! Please log in.")
                return True
            else:
                st.error("Username already exists")

    return False


def main_page():
    """
    Frontend part of Streamlit for Home/Landing page.

    """

    # Load the model to activate
    model = load_model()

    def set_bg_from_url():
        """
        To unpack an image from url and set as background image

        """
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background: url("https://mir-s3-cdn-cf.behance.net/project_modules/disp/9c0722106004343.5f85fead2894a.gif");
                 background-size: 100% 100%;
                 background-repeat: no-repeat;
             }}
             </style>
             """,
            unsafe_allow_html=True
        )

    def sidebar_bg(side_bg):
        """
        To set a background image for the web app's sidebar
        Param: Image to be set as background

        """
        side_bg_ext = 'gif'

        st.markdown(
            f"""
          <style>
          [data-testid="stSidebar"] > div:first-child {{
              background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
              background-size: 90% 45%;
              background-repeat: no-repeat;
              background-position: bottom;
          }}
          </style>
          """,
            unsafe_allow_html=True,
        )

    side_bg = 'media/face4.gif'
    sidebar_bg(side_bg)

    set_bg_from_url()

    # Sidebar design logic
    with st.sidebar:
        # Set Sidebar Content
        st.sidebar.image('media/logo.png', use_column_width=True, output_format='auto')

        st.markdown("""
                    <style>
                        /* Define the animation */
                        @keyframes levitate {
                                0% { transform: translateY(0); }
                                50% { transform: translateY(-10px); }
                                100% { transform: translateY(0); }
                        }
                        /* Apply the animation to the image */
                        img {
                                animation: levitate 2s infinite;
                        }
                    </style>
                """, unsafe_allow_html=True)

        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("## Navigation")
        page = st.sidebar.selectbox("Go to:", ("🏠 Home", "⚙ Architecture"))
        if st.session_state.get('username'):
            st.button("Logout", on_click=logout)
        st.sidebar.write("")

    # Home page design
    if page == "🏠 Home":
        st.title('🔍 Truth-Scan v1.3')

        # Short introduction to model
        st.markdown('''<span style="font-size:20px;">
        Our web app named Truth-Scan harnesses the power of a CNN based model named EfficientNetB4.
        <br>It is a light-weight model used for image classification tasks made using Tensorflow as the framework.</br>
        </span>''', unsafe_allow_html=True)
        colA, colB = st.columns(2)

        # Web app usage instructions
        colA.markdown('''<span style="font-size:20px;">
            How to use this WebApp?
            <br>1. Upload an image.
            <br>2. Adjust the threshold.
            <br>3. Image processing will take place and the appropriate classification result will be displayed with score.</br>            
        </span>''', unsafe_allow_html=True)
        colB.image('media/icon.png')

        # Model limitations
        st.caption('<span style="font-size:20px;">Note: Please familiarize yourself with the limitations before utilizing the model.</span>', unsafe_allow_html=True)
        with st.expander('Limitations', expanded=False):
            st.markdown('''<span style="font-size:20px;">
                1. Variations in lighting, image quality, and facial expressions may impact the model's performance.
                <br>2. The model may encounter challenges in detecting emerging deepfake techniques, which utilize advanced descriptors to create highly realistic images.
                <br>3. Deepfakes created with 3D image processing or manual modifications may not be accurately recognized by the model.</br>
            </span>''', unsafe_allow_html=True)

        # File uploader and deterministic threshold adjusting slider for input image analysis
        img_uploaded = st.file_uploader("Upload an image...", type=["jpg", "png"])
        threshold = st.select_slider('Threshold', options=[i / 100 for i in range(0, 101, 5)], value=0.5)

        # Image analysis and classification logic
        if img_uploaded is not None:
            with st.spinner('Processing the image, getting faces...'):
                image = cv2.imdecode(np.frombuffer(
                    img_uploaded.read(), dtype=np.uint8), 1)
                face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                st.warning('Faces not found!')
            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]
                with st.spinner(f'Preprocessing the face #{i + 1}:'):
                    processed_face = preprocess_image(face_image)
                    processed_face = np.expand_dims(processed_face, axis=0)

                    prediction = model.predict(processed_face)

                    predicted_class = "FAKE" if prediction[0,
                    0] > threshold else "REAL"

                    # Classification result with confidence score
                    st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB),
                             caption=f"Face {i + 1}: {predicted_class} | Score: {prediction[0, 0]:.2f}", width=350)

                    download_img = cv2.imencode('.png', face_image)[1].tobytes()
                    st.download_button(label="Download Image", data=download_img,
                                       file_name=f"face_{i + 1}_{predicted_class}.png", mime="image/png")

                    # Prediction result
                    if predicted_class == "FAKE":
                        st.warning('The image is most likely fake!')
                    else:
                        st.success('The image is most likely real!')

                    st.divider()

    # Architecture page design
    elif page == "⚙ Architecture":
        st.title("Architecture")
        st.image('media/architecture_3.JPG', use_column_width=True)

        # About model's architectural layer functionalities
        st.markdown(
            """
            <div style="text-align: justify; font-size: 20px;">
                <h3>Working:</h3>
                <ol style="font-size: 20px;">
                    <li><strong>Input Image:</strong> The process starts with an input image that you want to classify. It is the raw image file that you provide to the model.</li>
                    <li><strong>CV2 Preprocessing:</strong> This is the first stage of processing where the image is converted into a numpy matrix. CV2 likely refers to OpenCV, a library of programming functions mainly aimed at real-time computer vision. This step involves resizing the image to the dimensions required by the model, normalizing the pixel values, etc.</li>
                    <li><strong>EfficientNet Preprocessing:</strong> This step involves specific preprocessing required by the EfficientNet architecture. It could include scaling pixel values in a certain way or applying other transformations that the EfficientNet model expects.</li>
                    <li><strong>EfficientNetB4 Model Architecture Layers:</strong> This is the core of the architecture where the image goes through several convolutional layers. These layers are designed to extract features from the image at various levels of abstraction.</li>
                    <li><strong>MBConv:</strong> Stands for Mobile Inverted Bottleneck Conv. These are a series of layers that use depthwise separable convolutions which are efficient both in terms of computation and memory. The numbers 3x3 or 5x5 refer to the size of the filters used in the convolution layers. A 3x3 filter will look at a 3 pixel by 3 pixel area of the image at a time, for example.</li>
                    <li><strong>GlobalAveragePooling2D:</strong> This layer reduces the spatial dimensions (i.e., width and height) of the input feature map to a single vector per map. This is useful to condense the information extracted by the convolutional layers into a form that can be used for classification.</li>
                    <li><strong>Dense (Sigmoid):</strong> This is a fully connected layer that uses the sigmoid activation function. It outputs a probability between 0 and 1, indicating the likelihood of the input image being classified as 'fake' or 'real'.</li>
                    <li><strong>Output:</strong> The final output is a single scalar value. If the value is closer to 1, the model predicts that the image is 'fake'. If the value is closer to 0, the model predicts that the image is 'real'.</li>
                </ol>
             </div>
           """, unsafe_allow_html=True)

# Main method/driver code
def main():
    if not authentication_page():
        return

    main_page()


if __name__ == "__main__":
    main()
