from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load pre-trained MobileNetV2 (ImageNet weights)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)  # 2 classes: Mango / Not Mango

model = Model(inputs=base_model.input, outputs=preds)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data pipeline
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory('dataset', target_size=(224,224),
                                              batch_size=32, subset='training')
val_gen = train_datagen.flow_from_directory('dataset', target_size=(224,224),
                                            batch_size=32, subset='validation')

# Train
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
model.save('mango_model.h5')
