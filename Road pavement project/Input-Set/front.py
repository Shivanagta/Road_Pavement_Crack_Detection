import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to process the image
def process_image():
    # Read the selected image
    img = cv2.imread(image_path.get())

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image processing ( smoothing )
    # Averaging
    blur = cv2.blur(gray,(3,3))

    # Apply logarithmic transform
    img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

    # Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)

    # Image smoothing: bilateral filter
    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

    # Canny Edge Detection
    edges = cv2.Canny(bilateral,100,200)

    # Morphological Closing Operator
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Create feature detecting method
    orb = cv2.ORB_create(nfeatures=1500)

    # Make featured Image
    keypoints, descriptors = orb.detectAndCompute(closing, None)
    featuredImg = cv2.drawKeypoints(closing, keypoints, None)

    # Display the processed image
    img_display.configure(image=featuredImg)

# Create the main window
window = tk.Tk()

# Create a label for the image
img_label = tk.Label(window, text="Select an image:")
img_label.pack()

# Create a variable to store the image path
image_path = tk.StringVar()

# Create a button to insert an image
insert_button = tk.Button(window, text="Insert Image", command=lambda: image_path.set(filedialog.askopenfilename()))
insert_button.pack()

# Create a button to process and display the image
process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.pack()

# Create a label to display the processed image
img_display = tk.Label(window)
img_display.pack()

# Run the main loop
window.mainloop()