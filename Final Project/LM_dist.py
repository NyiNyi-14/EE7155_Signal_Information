# %% Import lib
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import OrderedDict
from DnCNN import DnCNN
from DCT_IDCT import DCT_IDCT
import matplotlib.pyplot as plt
import pandas as pd

# %% Get work dir
import os
os.chdir("/Users/nyinyia/Documents/09_LSU_GIT/03_LM_dist")
print(os.getcwd())
print(os.listdir())

# %% Step1.1, Getting frames from video
video_path = '01_videos/03_deform.wmv'
output_folder = '02_frames/01_deform_frames/01_original' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_skip = 1 
cap = cv2.VideoCapture(video_path) 
fps = cap.get(cv2.CAP_PROP_FPS)  
start_time = 3  
end_time = 15 
start_frame = int(start_time * fps)  
end_frame = int(end_time * fps) 

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    if frame_count >= start_frame and frame_count <= end_frame:
        frame_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.png')

        if frame_count % frame_skip == 0:
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames from {start_time} to {end_time} seconds.")

# %% Step1.2, Grayscale + DCT + IDCT
block_size = 8
zero_center = True  # JPEG-like zero-centering
deblocking = DCT_IDCT(block_size = block_size)
DCT_folder = '02_frames/01_deform_frames/02_DCT'

if not os.path.exists(DCT_folder):
    os.makedirs(DCT_folder)
RGB_frames = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])

for frames in RGB_frames:
    RGB_path = os.path.join(output_folder, frames) 
    DCT_path = os.path.join(DCT_folder, frames) 
    img = cv2.imread(RGB_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    gray = gray.astype(np.float32)
    if zero_center: # JPEG-like
        gray -= 128  # Zero-center to [-128, 127]
    dct_img, h, w = deblocking.dct2_blockwise(gray) # DCT
    recon_img = deblocking.idct2_blockwise(dct_img, h, w) # IDCT
    if zero_center:
        recon_img += 128 
    recon_img = np.clip(recon_img, 0, 255).astype('uint8')
    cv2.imwrite(DCT_path, recon_img)

print(f"Total {len(RGB_frames)}: Filtered out high freq in DCT and reconstructed with IDCT.")

# %% Step1.3, DnCNN
model = DnCNN(channels=1, num_of_layers=17)
# state_dict = torch.load('models/net_blind.pth', map_location='cpu') # blind noise
# state_dict = torch.load('models/net_n025.pth', map_location='cpu') # n025 noise
state_dict = torch.load('models/net_n050.pth', map_location='cpu') # n050 noise
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

Denoised_folder = '02_frames/01_deform_frames/03_denoised'
if not os.path.exists(Denoised_folder):
    os.makedirs(Denoised_folder)

DCT_frames = sorted([f for f in os.listdir(DCT_folder) if f.endswith('.png')])

for frames in DCT_frames:
    noise_path = os.path.join(DCT_folder, frames) 
    denoised_path = os.path.join(Denoised_folder, frames) 
    img = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  

    with torch.no_grad():
        noise = model(img_tensor)
        denoised = img_tensor - noise

    denoised_img = denoised.squeeze().numpy()
    denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(denoised_path, denoised_img)

print(f"Total {len(DCT_frames)}: Denoised")

# %% Step1.4, CLAHE
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
enchanced_folder = '02_frames/01_deform_frames/04_contrast'
if not os.path.exists(enchanced_folder):
    os.makedirs(enchanced_folder)
denoised_file = sorted([f for f in os.listdir(Denoised_folder) if f.endswith('.png')])

for frames in denoised_file:
    input_path = os.path.join(Denoised_folder, frames)
    output_path = os.path.join(enchanced_folder, frames)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 
    enhanced = clahe.apply(img) 
    cv2.imwrite(output_path, enhanced) 

print(f"Total {len(denoised_file)} Contrast Enhanced.")

# %% Gtting Histoplot
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
hist_in = cv2.imread("02_frames/01_deform_frames/03_denoised/frame_0120.png", cv2.IMREAD_GRAYSCALE)
hist_out = clahe.apply(hist_in)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.hist(hist_in.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.6)
plt.title(r'$\mathrm{Original \ Histogram}$', fontsize = 16)
plt.xlabel(r'$\mathrm{pixel}$', fontsize = 16)
plt.ylabel(r'$\mathrm{density}$', fontsize = 16)
plt.xlim(0, 256)
plt.tick_params(axis='both', labelsize=14) 
plt.grid()

plt.subplot(1, 2, 2)
plt.hist(hist_out.ravel(), bins=256, range=[0, 256], color='green', alpha=0.6)
plt.title(r'$\mathrm{CLAHE \ Histogram}$', fontsize = 16)
plt.xlabel(r'$\mathrm{pixel}$', fontsize = 16)
plt.ylabel(r'$\mathrm{density}$', fontsize = 16)
plt.xlim(0, 256)
plt.tick_params(axis='both', labelsize=14) 
plt.grid()

plt.tight_layout()
# plt.show()
# plt.savefig('/Users/nyinyia/Documents/08_LSU_courses/03_EE7155_signal_processing/Final Project/00_images/hist.pdf', format='pdf', bbox_inches='tight')

# %%  Step2, Edge detection + morphology + contour finding
Morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
enchanced_file = sorted([f for f in os.listdir(enchanced_folder) if f.endswith('.png')])
edges_folder = '02_frames/01_deform_frames/05_edges'  
morph_folder = '02_frames/01_deform_frames/06_morph'
contour_folder = '02_frames/01_deform_frames/07_contours'

os.makedirs(contour_folder, exist_ok=True)
os.makedirs(edges_folder, exist_ok=True)
os.makedirs(morph_folder, exist_ok=True)

count = np.empty(len(enchanced_file), dtype=object)

for i, frames in enumerate(enchanced_file):
    input_path = os.path.join(enchanced_folder, frames)
    output_edge = os.path.join(edges_folder, frames)
    output_morph = os.path.join(morph_folder, frames)
    output_con = os.path.join(contour_folder, frames)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, Morphology_kernel) # dilation followed by erosion
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    droplets = []
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0: 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                droplets.append({
                    "contour": cnt,
                    "centroid": (cx, cy),
                    "area": area
                })
    count[i] = droplets
    for d in droplets:
        cv2.drawContours(color_img, [d["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(color_img, d["centroid"], 3, (0, 0, 255), -1)

    cv2.imwrite(output_edge, edges)
    cv2.imwrite(output_morph, binary)
    cv2.imwrite(output_con, color_img)

# %% Getting features + plots
d_loc = np.zeros((len(droplets), 2))
d_area = np.zeros((len(droplets), 1))

for i, d in enumerate(droplets):
    d_loc[i][0] = d['centroid'][0]
    d_loc[i][1] = d['centroid'][1]
    d_area[i] = d['area']


# %% plot
cmap = plt.get_cmap('tab20')  
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for idx, (x, y) in enumerate(d_loc):  
    color = cmap(idx % 20)  
    ax.plot(x, y, 'o', color=color, markersize=14)
    ax.text(x + 10, y+2, f'LM{idx+1}', color=color, fontsize=16, weight='bold')  # small offset to the right

ax.set_xlabel(r'$\mathrm{x \ (pixels)}$', fontsize = 16)
ax.set_ylabel(r'$\mathrm{y \ (pixels)}$', fontsize = 16)
ax.set_title('Labeled LM Droplet Centroids', fontsize = 16)
plt.tick_params(axis='both', labelsize=14) 
ax.invert_yaxis() 
ax.grid(True)
plt.tight_layout()
# plt.show()
# plt.savefig('/Users/nyinyia/Documents/08_LSU_courses/03_EE7155_signal_processing/Final Project/00_images/cent.pdf', format='pdf', bbox_inches='tight')

# %% dataframe
df = pd.DataFrame({
    'ID': [f'LM{i+1}' for i in range(len(d_loc))],
    'Centroid_X': d_loc[:, 0],
    'Centroid_Y': d_loc[:, 1],
    'Area (px²)': d_area.flatten()
})

print(df)

# %% features from first image 
f_0000 = cv2.imread("02_frames/01_deform_frames/06_morph/frame_0000.png", cv2.IMREAD_GRAYSCALE)
con_0000, _ = cv2.findContours(f_0000, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

drop_0000 = []
color_img = cv2.cvtColor(f_0000, cv2.COLOR_GRAY2BGR)
for i in con_0000:
    a_0000 = cv2.contourArea(i)
    if a_0000 > 0: 
        M_0000 = cv2.moments(i)
        if M_0000["m00"] != 0:
            cx_0000 = int(M_0000["m10"] / M_0000["m00"])
            cy_0000 = int(M_0000["m01"] / M_0000["m00"])
            drop_0000.append({
                "contour": i,
                "centroid": (cx_0000, cy_0000),
                "area": a_0000
            })

loc_0000 = np.zeros((len(drop_0000), 2))
area_0000 = np.zeros((len(drop_0000), 1))

for i, d in enumerate(drop_0000):
    loc_0000[i][0] = d['centroid'][0]
    loc_0000[i][1] = d['centroid'][1]
    area_0000[i] = d['area']

loc_0000[1:] = np.array((loc_0000[2], loc_0000[1]))
area_0000[1:] = np.array((area_0000[2], area_0000[1]))
fig, ax = plt.subplots(1,1, figsize=(8,6))
for i in loc_0000:
    ax.plot(i[1], i[0], 'ro',)

cmap = plt.get_cmap('tab20') 

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for idx, (x, y) in enumerate(loc_0000): 
    color = cmap(idx % 20)  
    ax.plot(x, y, 'o', color=color, markersize=14)
    ax.text(x + 10, y-5, f'LM{idx+1}', color=color, fontsize=16, weight='bold')  

ax.plot(d_loc[0][0], d_loc[0][1], 'o', color = 'green', markersize=14, 
        label = r'LM1 location before actuation',)
ax.plot(d_loc[2][0], d_loc[2][1], 'o', color = 'red', markersize=14,
        label = r'LM3 location before actuation',)
ax.plot(loc_0000[1][0], loc_0000[1][1], 'x', color = 'red', markersize = 16,)
ax.set_xlabel(r'$\mathrm{x \ (pixels)}$', fontsize = 16)
ax.set_ylabel(r'$\mathrm{y \ (pixels)}$', fontsize = 16)
ax.set_title('Motion of LM Droplet', fontsize = 16)
plt.tick_params(axis='both', labelsize=14) 
ax.invert_yaxis() 
ax.grid(True)
ax.legend(fontsize = 16)
plt.tight_layout()
# plt.show()
# plt.savefig('/Users/nyinyia/Documents/08_LSU_courses/03_EE7155_signal_processing/Final Project/00_images/motion.pdf', format='pdf', bbox_inches='tight')

# %%