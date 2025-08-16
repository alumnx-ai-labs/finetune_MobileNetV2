from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from pathlib import Path

print("=== Testing Your Trained Model ===\n")

# Load the trained model
try:
    model = load_model('teachable_machine_style.h5')
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: Could not find 'teachable_machine_style.h5'")
    exit()

def predict_image(image_path):
    """Predict a single image and return result"""
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        pred = model.predict(img_array, verbose=0)[0]
        
        # Get class names (same order as training)
        classes = ['mango_trees', 'not_mango_trees']
        result = classes[np.argmax(pred)]
        confidence = np.max(pred)
        
        return result, confidence
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def test_dataset_images():
    """Test on images from your dataset"""
    print("\n🧪 Testing on DATASET images:\n")
    
    # Test mango images (with .JPG extension)
    mango_path = "dataset/mango_trees"
    if os.path.exists(mango_path):
        print("📁 Testing MANGO images:")
        # Look for .JPG files (uppercase)
        mango_files = list(Path(mango_path).glob("*.JPG"))[:3]  # First 3 images
        
        for img_file in mango_files:
            result, confidence = predict_image(str(img_file))
            status = "✅" if "mango" in result else "❌"
            print(f"   {status} {img_file.name}: {result} ({confidence:.1%})")
    
    # Test non-mango images
    non_mango_path = "dataset/not_mango_trees"
    if os.path.exists(non_mango_path):
        print("\n📁 Testing NON-MANGO images:")
        # Look for .JPG files (uppercase)
        non_mango_files = list(Path(non_mango_path).glob("*.JPG"))[:3]  # First 3 images
        
        for img_file in non_mango_files:
            result, confidence = predict_image(str(img_file))
            status = "✅" if "not_mango" in result else "❌"
            print(f"   {status} {img_file.name}: {result} ({confidence:.1%})")

def test_dedicated_test_folder():
    """Test on images in your 'test' folder"""
    print("\n🎯 Testing on TEST FOLDER images:\n")
    
    test_path = "test"
    if os.path.exists(test_path):
        # Look for all image files in test folder
        image_extensions = ['.JPG', '.jpg', '.jpeg', '.png', '.PNG']
        test_files = []
        
        for ext in image_extensions:
            test_files.extend(list(Path(test_path).glob(f"*{ext}")))
        
        if test_files:
            print("📁 Testing images in test folder:")
            for img_file in test_files:
                result, confidence = predict_image(str(img_file))
                print(f"   🔮 {img_file.name}: {result} ({confidence:.1%})")
        else:
            print("❌ No image files found in test folder")
    else:
        print("❌ Test folder not found")

def show_dataset_info():
    """Show info about your dataset"""
    print("📊 Your Dataset Info:")
    print("-" * 40)
    
    # Count mango images
    mango_path = "dataset/mango_trees"
    mango_count = len(list(Path(mango_path).glob("*.JPG"))) if os.path.exists(mango_path) else 0
    
    # Count non-mango images  
    non_mango_path = "dataset/not_mango_trees"
    non_mango_count = len(list(Path(non_mango_path).glob("*.JPG"))) if os.path.exists(non_mango_path) else 0
    
    # Count test images
    test_path = "test"
    test_count = len([f for f in os.listdir(test_path) if f.endswith(('.JPG', '.jpg', '.png'))]) if os.path.exists(test_path) else 0
    
    print(f"🥭 Mango images: {mango_count}")
    print(f"🌿 Non-mango images: {non_mango_count}")
    print(f"🧪 Test images: {test_count}")
    print(f"📈 Total training: {mango_count + non_mango_count}")

# Main execution
if __name__ == "__main__":
    show_dataset_info()
    
    print("\nChoose test mode:")
    print("1. Test on dataset samples")
    print("2. Test on dedicated test folder") 
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_dataset_images()
    elif choice == "2":
        test_dedicated_test_folder()
    elif choice == "3":
        test_dataset_images()
        test_dedicated_test_folder()
    else:
        print("Invalid choice. Running both tests...")
        test_dataset_images()
        test_dedicated_test_folder()
    
    print("\n✅ Testing complete!")