import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def cnn_train_model():
    # -------------------------------
    # Load the Data
    # -------------------------------
    X_train = np.load('ml_data/patch_size25_frames_10_ref_test/train_data.npy')
    y_train = np.load('ml_data/patch_size25_frames_10_ref_test/train_targets.npy')

    X_val = np.load('ml_data/patch_size25_frames_10_ref_test/val_data.npy')
    y_val = np.load('ml_data/patch_size25_frames_10_ref_test/val_targets.npy')

    X_test = np.load('ml_data/patch_size25_frames_10_ref_test/test_data.npy')
    y_test = np.load('ml_data/patch_size25_frames_10_ref_test/test_targets.npy')

    # -------------------------------
    # Preprocess the Data
    # -------------------------------
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # -------------------------------
    # Build the CNN Model
    # -------------------------------
    model = Sequential([
        # First convolution block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
        AveragePooling2D(pool_size=(2, 2)),
        
        # Second convolution block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        AveragePooling2D(pool_size=(2, 2)),
        
        # Third convolution block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        AveragePooling2D(pool_size=(2, 2)),
        
        # Flatten and add Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model.
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Print a summary of the model architecture.
    model.summary()

    # -------------------------------
    # Train the Model
    # -------------------------------
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_val, y_val)
    )

    # -------------------------------
    # Evaluate the Model
    # -------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

    # -------------------------------
    # Create directory and Save the Model
    # -------------------------------
    save_dir = "cnn_model_parameters"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "my_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
