# team-82-FormCorrect

A simple web app using computer vision to correct and provide feedback based on a user provided ground truth. 

Uses a expo-react app and a Python Flask backend to serve the user with a text and graph based response. 

Served using a web app, so installation is just opening the website. 

To run backend, you will need npm, and python. 

Set-up a conda env using the .environment file to run the python file. 

# Technical Analysis
![image](https://github.com/user-attachments/assets/e335aa20-0083-4296-8856-ee6a9f244d10)

# Developers 
Chi Jay Xu: UI and Web-app development
Dhruv Kapadia: Front-end and API management
Soohwan Kim: Computer-Vision based coordinate analysis
Aaron Chen: Coordinate and data analysis


















## description 
    We have so far taken a video via file path (local) and used OpenCV to open it and process each frame. We had to do some manual rotations because OpenCV does not track rotational aspects provided in metadata—or some videos simply don’t include it for some reason (which accounts for the bulk of the code). We used the Task API version of MediaPipe to directly load the pre-trained model. (It turns out the Solutions API for MediaPipe is more beginner-friendly, but we saw it later and didn’t use it.) We extracted the world coordinates per frame, stored them in a list, and converted it into a NumPy array at the end. This is just the data preparation step for the actual comparison between datasets that we can use—either to feed into a neural network or for a more direct comparison method like calculating joint angles and similar metrics. (The only downside of calculating manually is that we’ll need a classification algorithm to identify which exercise it is before we can assess how “correct” it is.) 

    From "recursiveregressionmodel.py" - the model has 4 main functions: actual_model, actual_model_modified, plot_output, and fake_actual_model
    - actual_model takes in two addresses to access two local mp4 files. Once taken the function outputs an array of strings and a tuple of information - the tuple contains the overall distance between points in the model, rotation on x axis, rotation on y axis, rotation on z axis, and point that is axis of rotation
    - actual_model_modified is the same as actual model but it takes in world coordinates from pose.py
    - fake_actual_model has a prebuild input set for testing purposes and acts as actual_model_modified 
    - plot_output plots the iniial video, user_video, and modified user video overlayed on each other through mediapipe's world coordinates and the models outputs
f
## Envirenment setup 
    We use python 3.11.11 because mediapipe doesnt work with the latest version of 3.13
    Create Conda virtual env
        conda env create -f environment.yml
    Activate the virtual env
        conda activate myenv
    install dependencies
        pip install -r requirements.txt

    Frontend:
    - install Node.js if not installed already (v22.14.0 used)
    - first run "npm install" in bash
    - cd into ".\AppTesting\"
    - run "npm start" in bash to start up expo server

    Backend:
    - open another terminal
    - cd into the main ".\team-82-FormCorrect\" folder
    - activate conda env
    - run "python fileTesting.py" to start up the backend flask server
    - recursiveregression.py requires networkx 3.4.2 
        pip install networkx
    


