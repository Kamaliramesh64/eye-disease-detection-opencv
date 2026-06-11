import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread('/content/fundus.jpg')
if image is None:
    raise ValueError("Image not found")

image = cv2.resize(image, (512, 512))
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# PREPROCESSING (CLAHE)
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(2.0, (8,8))
enhanced = clahe.apply(gray)

# -----------------------------
# BLOOD VESSEL SEGMENTATION
# -----------------------------
kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_v)

vessels = cv2.adaptiveThreshold(
    tophat, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

vessel_density = cv2.countNonZero(vessels) / (512 * 512)

# -----------------------------
# BRIGHT LESION DETECTION
# -----------------------------
_, bright = cv2.threshold(enhanced, 160, 255, cv2.THRESH_BINARY)

kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
bright_clean = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel_b)

contours, _ = cv2.findContours(bright_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lesion_count = 0
lesion_area = 0
final_output = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 80 < area < 1200:   # valid lesion range
        lesion_count += 1
        lesion_area += area
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(final_output, (x,y), (x+w,y+h), (0,255,255), 2)

# -----------------------------
# FINAL DECISION
# -----------------------------
if vessel_density < 0.15:
    result = "Defected Fundus Image"
elif lesion_count >= 8 and vessel_density < 0.30:
    result = "Defected Fundus Image"
else:
    result = "Normal Fundus Image"

# -----------------------------
# DISPLAY ALL OUTPUT IMAGES
# -----------------------------
plt.figure(figsize=(12,10))

plt.subplot(2,3,1)
plt.imshow(rgb)
plt.title("Original Fundus Image")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(enhanced, cmap='gray')
plt.title("Preprocessed (CLAHE)")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(bright_clean, cmap='gray')
plt.title("Bright Lesion Threshold Image")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(vessels, cmap='gray')
plt.title("Blood Vessel Segmentation")
plt.axis('off')
          plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.title("Detected Bright Lesions")
plt.axis('off')
plt.subplot(2,3,6)
plt.text(0.1, 0.6,
         f"Vessel Density : {vessel_density:.3f}\n"
         f"Lesion Count   : {lesion_count}\n"
         f"Lesion Area    : {lesion_area:.1f}\n\n"
         f"Final Output:\n{result}",
         fontsize=11)
plt.axis('off')
plt.tight_layout()
plt.show()
