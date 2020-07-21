import shutil
import os

def delete_folders(folder_to_delete):
	try:
		shutil.rmtree(folder_to_delete)
	except:
		print("Error while deleting directory {}".format(folder_to_delete))

def delete_files(file_to_delete):
	try:
		os.remove(file_to_delete)
	except:
		print("Error while deleting file {}".format(file_to_delete))
