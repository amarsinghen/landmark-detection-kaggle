from PIL import Image, ImageOps

def process_save_images(image_path):
    local_location = image_path.replace("raw_data", "set_224")
    # print(image_path)
    # print("local location1 : {}".format(local_location1))
    image = Image.open(image_path)
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    image.save(local_location)
