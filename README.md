
# Dataset

LSA64: A Dataset for Argentinian Sign Language
The sign database for the Argentinian Sign Language, created with the goal of producing a dictionary for LSA and training an automatic sign recognizer, includes 3200 videos where 10 non-expert subjects executed 5 repetitions of 64 different types of signs. Signs were selected among the most used ones in the LSA lexicon, including both verbs and nouns.

![LSA64](/images/LSA64.png)  

### Signs:
![Signs](/images/signsTable2.png)

### links      
[LSA64 website](https://facundoq.github.io/datasets/lsa64/)     
model was trained on the raw version:     
[drive link](https://drive.google.com/file/d/1C7k_m2m4n5VzI4lljMoezc-uowDEgIUh/view?usp=sharing)      
[mega link](https://mega.nz/#!kJBDxLSL!zamibF1KPtgQFHn3RM0L1WBuhcBUvo0N0Uec9hczK_M)

 
# Preprocessing
### 1. Resampling      
Video is a 4D array RGP and the fourth dimension is the number of frames so we can think of a video as an array of frames “ignoring audio”.
Not all cameras have the same fps “frames per seconds”, and not all words take the same amount of time to be done.
so we need to choose a fixed number of frames for each video and after trial and error we found that 10 fps and 2 seconds per word yields the best accuracy.

### 2. Pose detection “media pipe.”      
Pose detection from images is a computer vision technique that aims to identify and locate the key body joints of a person in an image.
We used Media Pipe holistic model to get pose information for each frame in video, Media Pipe holistic model is Real-time, simultaneous perception of human pose, face landmarks and hand tracking.       
![mediapipe js](/images/poses.gif)  


### 3. Feature extraction   
![GRU](/images/GRU.png)   
GRU stands for Gated Recurrent Unit. It is a type of recurrent neural network (RNN) that was introduced by Cho et al. in 2014 as a simpler alternative to Long Short-Term Memory (LSTM) networks. Like LSTM, GRU can process sequential data such as text, speech, and time-series data.

### 4. Classification   
For classification we take the output from GRU as the input for the SoftMax layer which output a probability distribution for all the words in the data set then we take the word with the highest probability as the final output

# Results
Model scored 97% on train set (70% of the data).        
95% on validation set (15% of the data).       
And 94% on test set (15% of the data).       
We also created our own small test data set and achieved 98% accuracy.      

# Demo   "click img"

<!-- ![Demo](/images/demo.mp4)    -->


[![Video Demo](/images/demoimg.png)](https://drive.google.com/file/d/1wJybd5MzzDjvircOuFtZM-okbDvm2hsn/view?usp=sharing)
