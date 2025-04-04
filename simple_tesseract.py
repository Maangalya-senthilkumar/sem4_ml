# Install dependencies
!apt-get install -y tesseract-ocr
!pip install pytesseract
!pip install pillow

import pytesseract
from PIL import Image
from google.colab import files

# Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Open and process the image
image = Image.open(image_path)
text = pytesseract.image_to_string(image)

# Display the extracted text
print("Extracted Text:\n", text)
