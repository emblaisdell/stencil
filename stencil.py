import cv2
import numpy as np

# Load your image (replace 'your_image.png' with your actual image file)
image = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)

# Ensure the image is binary (black and white)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find all connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# Get image dimensions
height, width = image.shape[:2]

# Create a mask to filter out components touching the border
border_mask = np.zeros(labels.shape, dtype=np.uint8)
border_mask[0:1, :] = 255  # Top border
border_mask[:, 0:1] = 255  # Left border
border_mask[-1:, :] = 255  # Bottom border
border_mask[:, -1:] = 255  # Right border

# Filter out components that touch the border
filtered_components = []
for label in range(1, num_labels):  # Start from 1 to ignore background (label 0)
    x, y, w, h, area = stats[label]
    
    # Check if the bounding box of the component intersects with the border mask
    if np.any(np.logical_and(border_mask[y:y+h, x:x+w] > 0, labels[y:y+h, x:x+w] == label)):
        continue  # Skip components touching the border
    
    # Add the component to the list of filtered components
    filtered_components.append(centroids[label])

# Display or process the filtered components as needed
for [x,_] in filtered_components:
    # Do something with each component, for example, display it
    cv2.line(image, (int(x), 0), (int(x),  height), (255, 255, 255), 10)

cv2.imwrite("output_image_with_supports.png", image)
