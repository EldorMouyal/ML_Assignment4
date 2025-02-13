import os
import numpy as np
import scipy.io
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Importing TensorFlow for VGG19
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# LOAD DATA LOCALLY
# Path to the labels file
labels_path = 'imagelabels.mat'
# Load the labels
labels_data = scipy.io.loadmat(labels_path)
labels = labels_data['labels'][0]  # Extract the labels array
# Directory containing images
image_dir = '102flowers/jpg'
# Get list of image files and sort them
image_files = sorted(os.listdir(image_dir))

# Sanity check
if len(image_files) != len(labels):
    print("check your data, labels does not match images file.")
    print(f"Number of images: {len(image_files)}")
    print(f"Number of labels: {len(labels)}")

# CREATING DATAFRAME
# Create a DataFrame for easy data handling
data = pd.DataFrame({
    'filename': image_files,
    'label': labels
})
num_classes = len(np.unique(data['label']))

# Function to verify images
def verify_images(df, image_dir):
    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        try:
            img = Image.open(img_path)
            img.verify()  # Verifies that the file is an image
            img.close()
        except (IOError, SyntaxError) as e:
            print(f"Corrupt image file detected: {img_path}")
            # Handle the corrupt image accordingly (e.g., remove from dataset)


verify_images(data, image_dir)
# Assume 'data' DataFrame from previous steps is already available
# 'data' has two columns: 'filename' and 'label'

# Split into 75% (train+val) and 25% test
train_val_data, test_data = train_test_split(
    data,
    test_size=0.25,
    stratify=data['label'],
    random_state=42
)

print(f"Training+Validation set size: {len(train_val_data)}")
print(f"Test set size: {len(test_data)}")


def split_train_validation(data, validation_size=0.333, random_state=None):
    """
    Splits the data into training and validation sets.
    Args:
        data (DataFrame): The dataset to split.
        validation_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    Returns:
        train_data (DataFrame): Training set.
        val_data (DataFrame): Validation set.
    """
    train_data, val_data = train_test_split(
        data,
        test_size=validation_size,
        stratify=data['label'],
        random_state=random_state
    )
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True)


# Since train_val_data is 75% of the original data, a validation_size of 0.333 gives ~50/25 split
train_data, val_data = split_train_validation(train_val_data, validation_size=0.333, random_state=42)

print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# print(tf.__version__)
def run_vgg19(train_data, val_data, test_data, image_dir, num_classes=102, batch_size=32, epochs=10):
    """
    :param train_data: Training dataset.
    :param val_data: Validation dataset.
    :param test_data: Test dataset.
    :param image_dir: Directory where images are stores.
    :param num_classes: Number of categories/classes/
    :param batch_size: Batch size for training.
    :param epochs: Number of training epochs.
    :return:
        model: Trained Keras model.
        test_probabilities (DataFrame): DataFrame containing filenames and predicted probabilities.
    """
    # Image dimensions expected by VGG19
    img_height, img_width = 244, 244

    # Data augmentation and normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
    )

    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Creating data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=image_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_data,
        directory=image_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=image_dir,
        x_col='filename',
        y_col=None,  # We won't provide labels here
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    # Load the VGG19 model pre-trained on ImageNet, excluding the top layers
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Freeze the last 4 layers of the model
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Define the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Predict probabilities on the test set
    test_generator.reset()  # Ensure generator is at the start
    predictions = model.predict(test_generator, verbose=1)

    # Create a DataFrame with filenames and their predicted probabilities
    test_probabilities = pd.DataFrame({
        'filename': test_generator.filenames,
        'probabilities': list(predictions)
    })

    return model, test_probabilities

# Adjust labels in the DataFrames
def adjust_labels(_data):
    _data = data.copy()
    data['label'] = data['label'] - 1  # Adjust labels to start from 0
    data['label'] = data['label'].astype(str)  # Convert labels to strings
    return data


train_data = adjust_labels(train_data)
val_data = adjust_labels(val_data)
test_data = adjust_labels(test_data)

# Run the VGG19 model
model, test_probabilities = run_vgg19(
    train_data,
    val_data,
    test_data,
    image_dir,
    num_classes=num_classes,
    batch_size=32,
    epochs=30
)
# Display the first few entries
print(test_probabilities.head())

# Add a column for the predicted class
test_probabilities['predicted_class'] = test_probabilities['probabilities'].apply(lambda x: np.argmax(x))

# Optionally, adjust predicted_class to match original labels (add 1)
test_probabilities['predicted_label'] = test_probabilities['predicted_class'] + 1

print(test_probabilities[['filename', 'predicted_label']].head())







