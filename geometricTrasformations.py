import matplotlib.pyplot as plt

import numpy as np
import requests
import cv2

urls = [
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/barbara.png"
]

for url in urls:
    response = requests.get(url)
    filename = url.split("/")[-1]

    with open(filename, 'wb') as f:
        f.write(response.content)
#loaded images

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()


#GEOMETRIC TRANSFORMATIONS
#scaling

#using the function resize() from cv2 module for this purpose. You can specify the scaling factor or the size of the image:

toy_image = np.zeros((6,6))
toy_image[1:5,1:5]=255
toy_image[2:4,2:4]=0
plt.imshow(toy_image,cmap='gray')
plt.show()
toy_image

#The parameter interpolation estimates pixel values based on neighboring pixels. INTER_NEAREST uses the nearest pixel and INTER_CUBIC uses several pixels near the pixel value we would like to estimate. INTER_LINEAR is the default interpolation method. The function resize() returns the scaled image.

toy_image = np.zeros((6,6))
toy_image[1:5,1:5]=255
toy_image[2:4,2:4]=0
plt.imshow(toy_image,cmap='gray')
plt.show()
toy_image


#new image
image = cv2.imread("lenna.png")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

#scale the horizontal axis by two and leave the vertical axis as is:
new_image = cv2.resize(image, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#scale vertical axis by two:
new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#We can scale the horizontal axis and vertical axis by two. The function resize() returns the scaled image.
new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#we can also shrink the image by setting the scaling factor to less than one:
new_image = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#specify the numbers and rows and columns of the new image:
rows = 100
cols = 200
new_image = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print("old image shape:", image.shape, "new image shape:", new_image.shape)

#TRANSLATION
#shifting image location in viewable area
tx = 100
ty = 0
M = np.float32([[1, 0, tx], [0, 1, ty]])
M

#shape of image is 
rows, cols, _ = image.shape

# warpAffine from the cv2 module. The first input parater is an image array, the second input parameter is the transformation matrix M, and the final input paramter is the length and width of the output image  (𝑐𝑜𝑙𝑠,𝑟𝑜𝑤𝑠)
 #here we shift the image 100 pixels horizontally to the right

new_image = cv2.warpAffine(image, M, (cols, rows))

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
# fix this by changing the output image size: (cols + tx,rows + ty):

new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

#We can shift the image horizontally:
tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
new_iamge = cv2.warpAffine(image, M, (cols + tx, rows + ty))
plt.imshow(cv2.cvtColor(new_iamge, cv2.COLOR_BGR2RGB))
plt.show()

#ROTATION
# We can rotate an image by angle θ which is achieved by the Rotation Matrix getRotationMatrix2D.

# center: Center of the rotation in the source image. We will only use the center of the image.

# angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).

# scale: Isotropic scale factor, in this course the value will be one.

# We can rotate our toy image by 45 degrees:

theta = 45.0
M = cv2.getRotationMatrix2D(center=(3, 3), angle=theta, scale=1)
new_toy_image = cv2.warpAffine(toy_image, M, (6, 6))

#plot the image
plot_image(toy_image, new_toy_image, title_1="Orignal", title_2="rotated image")

#intensity values
new_toy_image 


#lets do the same on color images
cols, rows, _ = image.shape

M = cv2.getRotationMatrix2D(center=(cols // 2 - 1, rows // 2 - 1), angle=theta, scale=1)
new_image = cv2.warpAffine(image, M, (cols, rows))

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image")
plt.show()


#MATHMATICAL OPERATIONS
#Array operations
#add a constant to intensity value

new_image = image + 20

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title="Adding 20"
plt.show()

#we can multiply each value by an intensity
new_image = 10 * image
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title = "Multiplying by 10"
plt.show()

#We can add the elements of two arrays of equal shape. In this example, we generate an array of random noises with the same shape and data type as our image.

Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8) 

Noise.shape

#We add the generated noise to the image and plot the result. We see the values that have corrupted the image:

new_image = image + Noise

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

plt.show()

#we can multiply noise to the image and plot the result. We see the values that have corrupted the image:

new_image = image * Noise

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title="Multiplying noise"
plt.show()

#we can subtract noise to the image and plot the result. We see the values that have corrupted the image:

new_image = image - Noise
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title="Subtracting noise"
plt.show()

#MATRIX OPERATIONS

