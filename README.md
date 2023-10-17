# Detecting Cringe Facial Expressions

## About the Project

The use of the word cringe to describe uncomfortable, awkward, and even embarrassing phenomena has become increasingly popular in Western society. Currently, there is limited research around the facial expression that is induced by modern cringe definitions, making it hard to distinguish from neighboring expressions. We address this by differentiating action units induced from cringe-worthy scenarios to other nearby social signals, such as pain and disgust. We collected dynamic samples of facial expressions as a result of cringe, pain, and disgust and trained a Support Vector Machine using this data.

## Built With
* Python
* FFMPEG
* OpenFace

### Libraries Used

https://scikit-learn.org/stable/index.html

https://pandas.pydata.org/

https://matplotlib.org/3.5.0/index.html

https://docs.python.org/3/library/glob.html

https://docs.python.org/3/library/sys.html

https://seaborn.pydata.org/generated/seaborn.heatmap.html

https://docs.python.org/3/library/statistics.html


## Getting Started

### Requirements

Since this code is saved as IPYNB files, the best way to run the code would be to download Jupyter Notebook and Python3. This allows the user to clearly distinguish different sections of the code and better understand the project in its entirety. 

### Structure

The three files that contain the project code are CringevsPain.ipynb, CringevsDisgust.ipynb, and CringevsPainvsDisgust.ipynb. The first of these files attempts to detect cringe facial expressions from pain facial expressions. Similarly, the second file compares cringe facial expressions with disgust facial expressions. Finally, the last file tries to detect between the three different types of facial expressions. 

### Data collection
The dynamic data has already been collected from sources, such as YouTube and Giphy. In addition, this data has been processed through OpenFace, which extracts features such as action units, from the videos. The dynamic samples were edited and cropped so that only one face is annotated per sample. The FeatureExtraction.ipynb is the code used to run samples through OpenFace. This file first installs OpenFace and then passes in the relevant mp4 files. This dataset is available in the Dataset(.csv) folder where the csv files are separated for each emotion. 

## Example Data
 
### Cringe

https://user-images.githubusercontent.com/51034700/163469103-f6d10550-39be-4373-b513-7fc15c4703a5.mp4

### Pain

https://user-images.githubusercontent.com/51034700/163468905-1bf3b816-e044-4f8d-a306-c955b7893f88.mp4

### Disgust

https://user-images.githubusercontent.com/51034700/163470990-0a39f743-2ea1-4f91-8bd7-8b1c74dbb8d4.mp4


## Self Evaluation

In our proposal, we aimed to detect social signals of cringe from social signals of nearby neighbors. Although we were able to compare it with pain and disgust, another important neighbour to consider would have been embarrassment. However, this additional social signal was skipped due to time constraints. Pain and disgust were prioritized, because they prominent social signals that we came accross when digging for cringe expressions. 

For the code and analysis portion, we used PCA, GMM, SVM and cross-validation, as set out in the proposal. When detecting cringe expressions against pain on its own, the accuracy and F1 scores were both ~60%. This is relatively poor accuracy when classifying two social signals, as most of the accurate results could be attributed purely to chance. When detecting cringe expressions against disgust, the accuracy and F1 scores were ~77%. This shows that cringe is very easily confused with pain, but can be distinguished from disgust. 

When classifying the three facial expressions in the same run, the test performance dropped to about ~50%. This was expected, given the direct comparisons. Digust was more easily differentiated, whereas pain and cringe were confused more often. In the future, a CNN model may be a way to expand this project and improve on its accuracy. 

