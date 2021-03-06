## Identifying 90,000 Important Landmarks in the World using Image Classification

Landmark recognition API takes an input image of any size then predicts and displays the Landmark Name out of 90k landmarks along with the confidence score.

### Business Use-case
Landmark recognition API helps travelers with finding the names of the historical places he/she is visiting by uploading the image of the building/landmark to the API.

### Running the FLASK API
- Download the repository.
- Ensure you have all the dependencies as in the requirements.txt file.
- Run app.py
- The API starts on the localhost:5000 and serves **/upload** and **/swagger** URIs

#### Flask API Architecture Diagram
![Screenshot](/model_weights/architecture_diagrams/flask_api_architecture.JPG)

### Data Pre-Processing and Modeling
All jupyter notebooks and python files for this step can be found under jupyter_notebooks folder in this project

#### Dataset
- The dataset was acquired from Google Landmark Recognition 2019 Kaggle challenge.
- The raw training dataset consisted of ~4 million images of different sizes with over 200k classes, total 500gb.

#### Hardware for this project
Intel 9900k 8 core cpu, 1 RTX 2080TI GPU, 64gb RAM

#### Model Training Architecture
![alt text](/model_weights/architecture_diagrams/model_training_architecture.JPG)

#### Data Pre-Processing
- Classes with 5 or less images were discarded since the dataset size is too small for those classes.
- The dataset was very impbalanced. There were over 200 important landmarks with >500 images. I decided to randomly sample only 500 images with these landmarks and got rid of rest.
- Now, there were ~6400 classes with >100 images and ~86000 classes with <=100 images. Therefore, the max number of images per class was set to 100 after DeLF feature matching step as described in next step. This helped with balancing the dataset somewhat.
- I used google's DeLF framework (https://www.tensorflow.org/hub/tutorials/tf_hub_delf_module) to match each image within a class and removed any image that had less than 11 inliers against other images in same class. This filered the noisy images and reduced the dataset size to ~2.2 million images with ~92k classes. This step took 2 days for extracting DeLF features with multiprocessing and 10 days for feature matching with multiprocessing.
- I also performed some ad-hoc analysis and removed some additional classes that mostly had people in them and no landmarks.
- All the images were re-sized to 224x224 RGB using PIL ImageOps.
- The dataset was very imbalanced since the range of images varied from 6 to 100 per class. Therefore, I decided to divide the dataset into 11 groups (based on number of images per class) for reducing the modeling time and train 11 models separately. For example, I grouped the dataset with 6 to 10 images per class as group1.
- The dataset was then split into train and validation datasets. The range of ratio of train/valid was 60/40 to 80/20 depending on the group. For group with smaller number of images per class had a lower train/valid ratio. 

#### Modeling
- Created a supervised multi-class classification model using Deep Learning Tensorflow 2 with Keras framework.
- 11 seperate models were trained as per the dataset division.
- Used transfer learning and Resnet50 with imagenet weights as the base network was used to train the models.
- Dropout layer and image augmentation techniques were used to help with severe overfitting.
- Training time for each model varied from 1 to 2 days depending on the number of images per group.
- To re-train the models, all the jupyter notebooks have been provided here, jupyter_notebooks\Models\...

#### Model Results
- The best validation accuracy score from the model training was 68%. There were several reasons for the overfitting that are listed in the challenges section.

#### Prediction
- The user input image is also resized to 224x224 RGB images with PIL ImageOps and scaled as in the training.
- Then we run the prediction against all models and present the top 2 results with name of the place to user based on the highest confidence score.

#### Challenges
- There were tremendous hardware limitations given the size of the dataset: 
  - For DeLF with multiprocessing, the CPU ran at near 100% capacity for days. 
  - Model training also took a couple days per group even with constant 90% cuda core usage in GPU during training.
- Tried to use google colab, but the dataset size was in 30gb range for 224x224 sized images. Bought extra space (2TB) on google drive to mount drive to google colab notebooks, but  trying to upload dataset with so many images to google drive was a big challenge.
- Also, since it tooks days to pre-process data and training, it is not feasable on google colab since it times out at max 24 hours, and if you consistenly use their resources, timeouts get more frequent with shorter time windows. Therefoe, ended up doing mostof the project on local desktop.
- Cleaning the dataset was also a big challange and thus really helped with learning new concepts to find best algorithms optimization strategies to process data.
- Tried various transfer learning models, VGG16, VGG19, Resnet50, Inception_V3, Inception_ResnetV2.
- Given the slowness in training, it was also challenging to do hyperparameter tuning since it took days to find best parameters.
- The model was consistently overfitting because of various reasons:
  - The number of images per class were really small, 6 minimum and 100 maximum. Most of the landmarks (~39k) had only 6 to 10 images, creating a lot of imbalance.
  - Even after DeLF step, there were a lot of images with only people, water, trees, bushes, no buildings etc., adding a lot of noise to dataset.
  - A lot of landmarks were just mountains and beaches. Therefore, this adds a lot of confusion to relate the exact beach or mountain to a specific landmark. I think even humans (as a true reference point) can wrongly identify these mountains and beaches.
  - After some more deep-dive analysis with LIME framework, I found that model was also learning blue sky as an important feature during training among many landmarks. Therefore, it added additional complexity in distinguishing different landmarks. 
  
#### Further Improvements/ Recommendations
- To process this scale of dataset, it is recommended to get a bigger hardware (Ideal in home relatively affordable: AMD Ryzen threadripper3990x 64-core, 4 RTX2080TI GPUs, 256GB RAM powered with solar panels :)). It will save days to tackle such a project.
- Further data cleaning using Object detection is recommended. We can remove images with only people, trees, water, and no buildings to reduce more noisy data. However, object detection per image took about ~1 sec on average. Given ~4 million images, it was not feasible with given hardware resources.
- We can also try to mask sky in the images since model was also recognizing on sky as a feature. I tried it, but again masking step took ~1-2 secs per image. Given the size of the dataset ~4 million images and hardware restrictions, it was not feasible to complete this step in a reasonable time frame.
