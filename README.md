##Google Landmark Recognition 2019 Kaggle 

Landmark recognition API takes an input image of any size and predicts and dislplays the landmark_id out of 92k landmarks along with the confidence score.

###Dataset
- The dataset was acquired from Google Landmark Recognition 2019 Kaggle challenge.
- The raw dataset consisted of ~4 million images with over 200k classes.

###Hardware for this project
Intel 9900k 8 core cpu, 1 RTX 2080TI GPU, 64gb RAM

###Data Pre-Processing
- Classes with 5 or less images were discarded since the dataset size is too small for those classes.
- The dataset was very impbalanced. There were over 200 important landmarks with >500 images. I decided to randomly sample only 500 images with these landmarks and got rid of rest.
- Now there were ~6400 classes with >100 images and ~86000 classes with <=100 images. Therefore, the max number of images per class was set to 100 after DeLF feature matching step as described in next step. This helped with balancing the dataset somewhat.
- I used DeLF to match each image within a class and removed any image that had less than 11 inliers against other images in same class. This filered the noisy images and reduced the dataset size to ~2.2 million images with ~92k classes. This step took 2 days for extracting DeLF features with multiprocessing and 10 days for feature matching with multiprocessing.
- I also performed some ad-hoc analysis and removed some additional classes that mostly had people in them and no landmarks.
- All the images were re-sized to 224x224 RGB using PIL ImageOps.
- Since the dataset with ~2.2 million is still huge, I decided to divide the dataset into 11 groups (based on number of images per class) for reducing the modeling time and train 11 models. This also helped with balancing the dataset since the range of images per class was 6 to 100. For example, I grouped the dataset with 6 to 10 images per class as group1.

###Modeling
- 11 seperate models were trained as per the dataset division and prediction was displayed to user based on the highest confidence score out of the 11 models for a landmark image.  
- Used transfer learning and Resnet50 with imagenet weights was used as the base network to train the model.
