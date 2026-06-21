import h5py
import numpy as np
from PIL import Image


# 1. Load the event file
file_path = '0F_s2eMM4VU_0.h5'  # Point this to your actual .h5 file!
with h5py.File(file_path, 'r') as f:
    # Adjust these keys if your simulator uses different internal names
    x = np.array(f['events/x'])
    y = np.array(f['events/y'])
    t = np.array(f['events/t'])
    p = np.array(f['events/p'])


# 2. Select a 30ms time window (assuming 't' is in microseconds)
# We start in the middle of the clip so we actually catch facial movement
t_start = t[-1] // 2 
t_end = t_start + 30000  # 30,000 microseconds = 30ms


mask = (t >= t_start) & (t <= t_end)
x_slice, y_slice, p_slice = x[mask], y[mask], p[mask]


# 3. Create a blank 188x188 image (White background)
height, width = 188, 188
img = np.ones((height, width, 3), dtype=np.uint8) * 255 


# 4. Color the pixels (Red for positive, Blue for negative)
for i in range(len(x_slice)):
    ix, iy = int(x_slice[i]), int(y_slice[i])
    # Ensure coordinates are within image bounds
    if ix < width and iy < height:
        if p_slice[i] == 1:
            img[iy, ix] = [255, 0, 0]   # Red
        else:
            img[iy, ix] = [0, 0, 255]   # Blue


# 5. Save the final image to plug into your LaTeX table
output_name = '0F_s2eMM4VU_0_events.png'
Image.fromarray(img).save(output_name)
print(f"Success! Event frame saved as {output_name}")
