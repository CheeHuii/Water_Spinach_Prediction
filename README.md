✅ Dataset:

Put your images in path/to/your/water_spinach_dataset structured like:

markdown
Copy
Edit
water_spinach_dataset/
    fresh/
    wilted/
    aging_dying/
    rotten/
Each folder contains ~200 images.

✅ Data Augmentation:

Applies random flips, rotations, zooms to diversify the dataset.

✅ Transfer Learning:

Uses MobileNetV2 pretrained on ImageNet for efficient feature extraction.

Adds a custom classification head with 4 classes.

✅ Training & Validation:

Splits data into 80% training and 20% validation.

✅ Test Prediction:

Uses a simple GUI (tkinter) to select an image from your system.

Displays the prediction and confidence.