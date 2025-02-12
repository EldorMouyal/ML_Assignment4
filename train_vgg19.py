import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def load_data_splits(split_dir):
    train_data = pd.read_csv(os.path.join(split_dir, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(split_dir, 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(split_dir, 'test_data.csv'))
    return train_data, val_data, test_data

def run_vgg19(train_data, val_data, test_data, image_dir, output_dir, num_classes=102, batch_size=32, epochs=10):
    # Ensure labels are strings (they should be from data_prep)
    # Image dimensions
    img_height, img_width = 224, 224

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=image_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    validation_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_data,
        directory=image_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=image_dir,
        x_col='filename',
        y_col=None,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    # Model setup
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'best_vgg19_model.h5'), save_best_only=True)
    ]

    # Training
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Load best model
    model.load_weights(os.path.join(output_dir, 'best_vgg19_model.h5'))

    # Predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)

    # Save predictions
    test_probabilities = pd.DataFrame({
        'filename': test_generator.filenames,
        'probabilities': list(predictions),
        'predicted_class': np.argmax(predictions, axis=1) + 1  # Adjust back to original labels
    })

    test_probabilities.to_csv(os.path.join(output_dir, 'vgg19_test_predictions.csv'), index=False)
    print("VGG19 training and prediction completed. Outputs saved to:", output_dir)

def main():
    # Paths
    split_dir = 'data_splits'
    image_dir = '102flowers/jpg'
    output_dir = 'vgg19_output'

    os.makedirs(output_dir, exist_ok=True)

    # Load data splits
    train_data, val_data, test_data = load_data_splits(split_dir)

    # Number of classes
    num_classes = len(train_data['label'].unique())

    # Run VGG19
    run_vgg19(train_data, val_data, test_data, image_dir, output_dir, num_classes=num_classes, epochs=30)

if __name__ == '__main__':
    main()
