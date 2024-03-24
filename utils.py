import cv2

def save_image_to_disk(output_dir, image):
    # If the function attribute does not exist, initialize it
    if not hasattr(save_image_to_disk, "counter"):
        save_image_to_disk.counter = 0  # Initialize the counter attribute
    
    # Define the image file name
    image_file_name = output_dir + "/image_" + str(save_image_to_disk.counter) + ".png"
    
    # Save the image
    cv2.imwrite(image_file_name, image)
    
    # Print the name of the file that was saved
    print("Saved: " + image_file_name)
    
    # Increment the counter for the next call
    save_image_to_disk.counter += 1