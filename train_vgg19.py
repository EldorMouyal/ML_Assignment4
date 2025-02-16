import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def load_data_splits(split_dir):
    train_data = pd.read_csv(os.path.join(split_dir, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(split_dir, 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(split_dir, 'test_data.csv'))
    return train_data, val_data, test_data

def run_vgg19(train_data, val_data, test_data, image_dir, output_dir, num_classes=102, batch_size=32, epochs=10):
    img_height, img_width = 224, 224

    train_data['label'] = train_data['label'].astype(str)
    val_data['label'] = val_data['label'].astype(str)
    test_data['label'] = test_data['label'].astype(str)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='reflect',
        brightness_range=[0.7, 1.3],  # Adjust brightness further
        channel_shift_range=50.0,  # Further color channel shifting
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
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    # Model setup
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add Dropout here
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freezing all layers except the last few
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Train the model again with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'best_vgg19_model.keras'), save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'vgg19_epoch_{epoch:02d}.keras'), save_freq='epoch')
    ]

    # Training
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Load best model
    model.load_weights(os.path.join(output_dir, 'best_vgg19_model.keras'))

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)

    # After training, you can call this function:
    plot_metrics(history, output_dir)

    # Save predictions
    test_probabilities = pd.DataFrame({
        'filename': test_generator.filenames,
        'probabilities': list(predictions),
        'predicted_class': np.argmax(predictions, axis=1) + 1  # Adjust back to original labels
    })

    test_probabilities.to_csv(os.path.join(output_dir, 'vgg19_test_predictions.csv'), index=False)
    print("VGG19 training and prediction completed. Outputs saved to:", output_dir)


def plot_metrics(history, output_dir):
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plots
    plt.savefig(os.path.join(output_dir, 'accuracy_loss_plots_VGG.png'))
    plt.close()

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
    run_vgg19(train_data, val_data, test_data, image_dir, output_dir, num_classes=num_classes, epochs=40)


if __name__ == '__main__':
    main()
