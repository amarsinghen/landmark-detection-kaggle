from PIL import Image, ImageOps

def process_save_images(image_path):
    local_location = image_path.replace("set_256", "set_128")
    # print(image_path)
    # print("local location1 : {}".format(local_location1))
    image = Image.open(image_path)
    image = ImageOps.fit(image, (128, 128), Image.ANTIALIAS)
    image.save(local_location)
