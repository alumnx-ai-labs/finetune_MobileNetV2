from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

print("=== Teachable Machine Style Training ===")
print("Two-stage approach: Head training → Fine-tuning\n")

# Load pre-trained MobileNetV2 (ImageNet weights)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom classification head with dropout (like Teachable Machine)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # Regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # More dropout before final layer
preds = Dense(2, activation='softmax')(x)  # 2 classes

model = Model(inputs=base_model.input, outputs=preds)

# Conservative data augmentation (less aggressive)
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,       # Reduced from 360
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,          # Reduced from 0.2
    width_shift_range=0.1,   # Small shifts
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],  # Less brightness variation
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory('dataset', target_size=(224,224),
                                              batch_size=16, subset='training')
val_gen = val_datagen.flow_from_directory('dataset', target_size=(224,224),
                                          batch_size=16, subset='validation')

print(f"Training images: {train_gen.samples}")
print(f"Validation images: {val_gen.samples}")
print(f"Classes: {list(train_gen.class_indices.keys())}")

# STAGE 1: Train only the head (freeze ALL base layers)
print("\n🎯 STAGE 1: Training classification head only...")
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001),  # Higher LR for head
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history1 = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

print(f"Stage 1 - Final accuracy: {max(history1.history['accuracy']):.3f}")

# STAGE 2: Fine-tune top layers ONLY
print("\n🔥 STAGE 2: Fine-tuning top layers...")
# Unfreeze only last 10 layers (more conservative than 20)
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00005),  # Much lower LR for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

print(f"Stage 2 - Final accuracy: {max(history2.history['accuracy']):.3f}")
print(f"Final validation accuracy: {max(history2.history['val_accuracy']):.3f}")

# Save model
model.save('teachable_machine_style.h5')
print("\n✅ Teachable Machine style model saved!")
print("🎯 Two-stage training completed:")
print("   Stage 1: Head training with frozen base")
print("   Stage 2: Fine-tuning with top 10 layers unfrozen")
print("🥭 Should now distinguish mango vs non-mango properly!")