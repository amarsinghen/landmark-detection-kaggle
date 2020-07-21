import tarfile
train_extract_location = "C:\\Apps\\jupyterWorkspace\\google_landmark_detection\\datasets\\train_raw"
for i in range(0,500):
    if i<10:
        train_tar_location = "D:\\Machine_Learning\\datasets\\google-landmark-detection\\tar_files\\images_00"+str(i)+".tar"
    elif (i>9 and i<100):
        train_tar_location = "D:\\Machine_Learning\\datasets\\google-landmark-detection\\tar_files\\images_0"+str(i)+".tar"
    else:
        train_tar_location = "D:\\Machine_Learning\\datasets\\google-landmark-detection\\tar_files\\images_"+str(i)+".tar"
    print(train_tar_location)
    try:
        my_tar = tarfile.open(train_tar_location)
        my_tar.extractall(train_extract_location)
    except tarfile.ReadError as err:
        print(err)
    finally:
        my_tar.close()