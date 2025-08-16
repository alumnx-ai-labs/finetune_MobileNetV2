from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from pathlib import Path

print("=== Quick Model Test ===")
print("Testing your trained model...\n")

# Load the trained model
try:
    model = load_model('teachable_machine_style.h5')
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: Could not find 'teachable_machine_style.h5'")
    print("Make sure you ran the training script first.")
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

def test_dataset_samples():
    """Test on some images from your dataset"""
    print("🧪 Testing on dataset samples:\n")
    
    # Test mango images
    mango_path = "dataset/mango_trees"
    if os.path.exists(mango_path):
        print("📁 Testing MANGO images:")
        mango_files = list(Path(mango_path).glob("*.jpg"))[:3]  # First 3 images
        
        for img_file in mango_files:
            result, confidence = predict_image(str(img_file))
            status = "✅" if "mango" in result else "❌"
            print(f"   {status} {img_file.name}: {result} ({confidence:.1%})")
    
    print()
    
    # Test non-mango images  
    non_mango_path = "dataset/not_mango_trees"
    if os.path.exists(non_mango_path):
        print("📁 Testing NON-MANGO images:")
        non_mango_files = list(Path(non_mango_path).glob("*.jpg"))[:3]  # First 3 images
        
        for img_file in non_mango_files:
            result, confidence = predict_image(str(img_file))
            status = "✅" if "not_mango" in result else "❌"
            print(f"   {status} {img_file.name}: {result} ({confidence:.1%})")

def test_single_image():
    """Test a single image path provided by user"""
    print("\n🎯 Single Image Test:")
    image_path = input("Enter the full path to an image file: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ File not found!")
        return
        
    result, confidence = predict_image(image_path)
    print(f"\n🔮 Prediction: {result} ({confidence:.1%})")

# Main execution
if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test on dataset samples (automatic)")
    print("2. Test single image (manual path)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        test_dataset_samples()
    elif choice == "2":
        test_single_image()
    elif choice == "3":
        test_dataset_samples()
        test_single_image()
    else:
        print("Invalid choice. Running dataset test...")
        test_dataset_samples()
    
    print("\n✅ Testing complete!")