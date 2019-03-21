# Pedestrian tracking 

The goal of that assignment was to develop a pedestrians tracking algorithm that did not use deep
 learning

### Requirements

You have to download the trained model and the data from https://www.dropbox.com/s/prjqkn40h0vvlae/pedestrian_tracking_data.zip?dl=0
 and copy it to `src` if you want to run the code.

### Algorithm developed

The algorithm developed is based on :
    
- Background subtraction  
- Pedestrian detection on detected motion

#### Pedestrian detection

To detect pedestrians we have implemented the algorithm described in 
`Histograms of Oriented Gradients for Human Detection` from Navneet Dalal and Bill Triggs and 
trained it on INRIA dataset http://lear.inrialpes.fr/data . 


#### Project structure
- `src/assignment2.ipynb` contains the description of the problem to solve.
- `src/model.py`contains the code to train or load the SVM trained to detect pedestrians
- `src/nannan.py` contains the code that given a directory containing the frames of a video 
returns bounding boxes for each pedestrian in each frame.
- `Report.pdf` is a summary of the developed algorithm. 


#### Evaluation

With the data in the above mentioned link this algorithm obtained 0.37 of Intersection over the 
Union for the detected bounding boxes.  