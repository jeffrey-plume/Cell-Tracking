### Intro
This notebook walks you through a script I developed for tracking cell movement using
segmentation, trajectory tracking, and video generation. 


## Data Source

This dataset, titled "PhC-C2DL-PSC" is part of the Cell Tracking Challenge and focuses on pancreatic stem cells cultured on a polystyrene substrate. It was provided by Dr. T. Becker and Dr. D. Rapoport from the Fraunhofer Institution for Marine Biotechnology in Lübeck, Germany. The dataset is part of the cell tracking challange, available [here](https://celltrackingchallenge.net/2d-datasets/).

First, I load the necessary libraries and initialize the cellpose model.  


```python
# Demonstrating the ability to manage data
import os
from cellpose import models
import pandas as pd
import trackpy as tp
import cv2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cellpose")

# Initialize Cellpose model
model = models.Cellpose(gpu=True, model_type='cyto')

```

Here, I load image data and prepare it for segmentation and tracking.



```python

# Initialize Cellpose model
model = models.Cellpose(gpu=True, model_type='cyto')

# Load .tif images as color
img_files = sorted(['02/' + f for f in os.listdir('02/') if f.endswith('.tif')])
imgs = [cv2.imread(x, cv2.IMREAD_COLOR) for x in img_files]

# Apply Cellpose (using channels 0 and 1)
masks, flows, styles, _ = model.eval(imgs, diameter=None, channels=[[0, 1]] * len(imgs))

print(f"Processed {len(imgs)} images with color input.")


```

    Processed 300 images with color input.
    


Using the Cellpose model, I perform segmentation to isolate cells within the image.  Below, I displayed the segmentation for every 40 images to visualize the cellpose results.



```python
import matplotlib.pyplot as plt
from cellpose import plot

nimg = len(imgs) 

# Loop through every tenth index
for idx in range(nimg)[::40]: 

    # Get the corresponding mask and flow
    maski = masks[idx]
    flowi = flows[idx][0]
    img = imgs[idx]
    # Plot segmentation results
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, maski, flowi)  
    plt.tight_layout()
    plt.show()

```


    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_0.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_1.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_2.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_3.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_4.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_5.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_6.png)
    



    
![png](PlumeJ%20coding%20example_files/PlumeJ%20coding%20example_6_7.png)
    


Next, I leverage Trackpy, a Python library for particle tracking, to connect cell centroids across multiple frames. I marked the cell centroids with a green circle and traced their path with a red line.


```python
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import trackpy as tp

# Function to get centroids from Cellpose masks
def get_centroids(masks):
    centroids = []
    unique_masks = np.unique(masks)  
    for region in unique_masks:
        if region == 0:  # Skip background
            continue
        coords = np.array(np.nonzero(masks == region)).T  
        centroid = coords.mean(axis=0)
        centroids.append((centroid[0], centroid[1]))  
    return centroids

gif_frames = []
tracking_list = []

# Process each frame
for k in range(nimg):
    # Get centroids
    centroids = get_centroids(masks[k])

    # Add tracking data to the list
    for c in centroids:
        tracking_list.append({'x': c[1], 'y': c[0], 'frame': k})

    # Convert tracking list to DataFrame at the end
    tracking_data = pd.DataFrame(tracking_list)

    # Link trajectories using Trackpy
    tracked = tp.link(tracking_data, search_range=10, memory=3)

    # Draw trajectories and centroids on the image
    img = imgs[k].copy() 
    for particle_id in tracked['particle'].unique():
        trajectory = tracked[tracked['particle'] == particle_id]
        points = trajectory[['x', 'y']].values
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i - 1]))
            pt2 = tuple(map(int, points[i]))
            cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # Red trajectory line

    # Draw current centroids
    for c in centroids:
        cv2.circle(img, (int(c[1]), int(c[0])), 3, (0, 255, 0), -1)  # Green circles
        
    # Convert frame to RGB for Pillow
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    pil_image = Image.fromarray(rgb_frame)
    gif_frames.append(pil_image)
    print(f"Processed frame {k}")
```

    Frame 299: 510 trajectories present.
    Processed frame 299
    

Lastly, I exported the frames to gif for viewing, visualizing every 10 frames due to file size limitations.


```python
# Save the GIF
output_gif = "tracked_cells_output.gif"
gif_frames[0].save(
    output_gif,
    save_all=True,
    append_images=gif_frames[::10], # every 10 frames for size
    duration=100,  
    loop=0  
)

print(f"GIF saved to {output_gif}")
```

    GIF saved to tracked_cells_output.gif
    

![Description](tracked_cells_output.gif)
