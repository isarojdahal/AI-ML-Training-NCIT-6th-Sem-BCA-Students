import tensorflow as tf
from tensorflow.keras import layers

# Step 1: Define parameters
image_size = (64, 64)  # Resize all images to this size
batch_size = 32
class_names = ['Flower', 'Bird', 'Human', 'Elephant', 'Car']  # Custom label order

# Step 2: Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "images",
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    class_names=class_names,
    shuffle=True
)

# Step 3: Split dataset into training and validation
val_batches = int(0.2 * len(dataset))
train_dataset = dataset.skip(val_batches)
val_dataset = dataset.take(val_batches)

# Optional performance optimizations
AUTOTUNE = tf.data.AUTOTUNE # constant in tf, which tells tf to automatically determine the optimal number of parallel operations based on system resources.
train_dataset = train_dataset.prefetch(AUTOTUNE) 
val_dataset = val_dataset.prefetch(AUTOTUNE) 
# prefetch() = While your model is training on one batch of data, TensorFlow prepares the next batch in the background. This helps keep the model running smoothly without waiting for data to be loaded.


# Step 4: Define a CNN model (no manual flattening or external normalization)
model = tf.keras.Sequential([
    layers.Input(shape=(image_size[0], image_size[1], 3)),
    layers.Rescaling(1./255), # for normalization.  1. for float division, 1 for int division.
    # It normalizes pixel values from the range [0, 255] to the range [0, 1] by dividing each pixel value by 255.
    layers.Conv2D(32, 3, activation='relu'), # 32 filters, 3x3 kernel, relu activation function.
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'), # 64 filters, 3x3 kernel, relu activation function.
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Save the model
model.save('mlp_model.keras')