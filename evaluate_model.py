import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
print("Loading model...")
model = load_model('mango_model.h5')

# Define class names (based on your folder structure)
class_names = ['mango_trees', 'not_mango_trees']

# Path to test images
test_folder = 'test'

# Get all image files from test folder
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
test_images = []

for file in os.listdir(test_folder):
    if any(file.endswith(ext) for ext in image_extensions):
        test_images.append(file)

if not test_images:
    print(f"No images found in '{test_folder}' folder!")
    print(f"Make sure you have images with extensions: {image_extensions}")
    exit()

print(f"Found {len(test_images)} test images")
print("-" * 50)

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# Function to predict and display results
def predict_image(img_path, filename):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get predicted class name
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

# Store results for summary
results = []

# Process each test image
print("EVALUATION RESULTS:")
print("=" * 60)

for i, filename in enumerate(test_images, 1):
    img_path = os.path.join(test_folder, filename)
    
    try:
        predicted_class, confidence, raw_predictions = predict_image(img_path, filename)
        
        # Store result
        results.append({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        })
        
        # Print result
        print(f"{i:2d}. {filename}")
        print(f"    Predicted: {predicted_class}")
        print(f"    Confidence: {confidence:.2f}%")
        print(f"    Raw scores: mango_trees={raw_predictions[0]:.4f}, not_mango_trees={raw_predictions[1]:.4f}")
        print()
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        print()

# Summary
print("=" * 60)
print("SUMMARY:")
print("-" * 60)

mango_count = sum(1 for r in results if r['predicted_class'] == 'mango_trees')
not_mango_count = sum(1 for r in results if r['predicted_class'] == 'not_mango_trees')

print(f"Total images tested: {len(results)}")
print(f"Predicted as mango_trees: {mango_count}")
print(f"Predicted as not_mango_trees: {not_mango_count}")
print()

# Show results by class
print("RESULTS BY PREDICTED CLASS:")
print("-" * 60)

print("\n🥭 MANGO TREES:")
mango_results = [r for r in results if r['predicted_class'] == 'mango_trees']
for r in mango_results:
    print(f"  • {r['filename']} ({r['confidence']:.1f}%)")

print("\n🌳 NOT MANGO TREES:")
not_mango_results = [r for r in results if r['predicted_class'] == 'not_mango_trees']
for r in not_mango_results:
    print(f"  • {r['filename']} ({r['confidence']:.1f}%)")

# Optional: Create a visual report (uncomment to enable)
def create_visual_report():
    """Create a matplotlib figure showing predictions"""
    if len(results) == 0:
        return
    
    # Calculate grid size
    n_images = min(len(results), 12)  # Show max 12 images
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        
        # Load and display image
        img_path = os.path.join(test_folder, results[i]['filename'])
        img = Image.open(img_path)
        
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        # Add title with prediction
        predicted = results[i]['predicted_class']
        confidence = results[i]['confidence']
        color = 'green' if predicted == 'mango_trees' else 'blue'
        
        axes[row, col].set_title(
            f"{results[i]['filename']}\n{predicted}\n({confidence:.1f}%)",
            fontsize=8, color=color, weight='bold'
        )
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visual report saved as 'evaluation_results.png'")
    plt.show()

# Uncomment the next line to generate visual report
create_visual_report()

print("\n✅ Evaluation complete!")