import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def cnn_train_model(identifier):
    # -------------------------------
    # Load the Data
    # -------------------------------
    X_train = np.load('ml_data/'+identifier+'/train_data.npy')
    y_train = np.load('ml_data/'+identifier+'/train_targets.npy')

    X_val = np.load('ml_data/'+identifier+'/val_data.npy')
    y_val = np.load('ml_data/'+identifier+'/val_targets.npy')

    X_test = np.load('ml_data/'+identifier+'/test_data.npy')
    y_test = np.load('ml_data/'+identifier+'/test_targets.npy')

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
    model_path = os.path.join(save_dir, identifier+"_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

def visualize_feature_maps(model_path, data_path, sample_index=0):
    """
    Visualize the feature maps of a saved CNN model after convolutional layers.
    
    Parameters:
    - model_path: Path to the saved model file (e.g., 'cnn_model_parameters/my_model.h5').
    - data_path: Path to the test data file (e.g., 'ml_data/'+identifier+'/test_data.npy').
    - sample_index: Index of the test image to visualize (default: 0).
    """
    
    # -------------------------------
    # Load the Model and Data
    # -------------------------------
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    model = load_model(model_path)
    print("Model loaded successfully.")
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return
    X_test = np.load(data_path)
    print("Test data loaded successfully.")
    
    # Select a sample image from the test set
    if sample_index >= len(X_test):
        print(f"Sample index {sample_index} is out of bounds.")
        return
    sample_image = X_test[sample_index]
    sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

    # -------------------------------
    # Get Feature Maps
    # -------------------------------
    feature_maps = model.predict(sample_image)

    # Print a summary of the model architecture.
    model.summary()

    # Assuming 'model' is your loaded/trained model
    first_conv_layer = model.layers[0]  # Get the first layer
    second_conv_layer = model.layers[1]  # Get the first layer
    third_conv_layer = model.layers[2]  # Get the first layer

    # Now pass the sample image through the first convolutional layer
    layer_output = first_conv_layer(sample_image)
    layer_output = second_conv_layer(layer_output)
    layer_output = third_conv_layer(layer_output)

    print("Output shape:", layer_output.shape)

    # -------------------------------
    # Visualize the Output (Feature Maps)
    # -------------------------------
    # Remove the batch dimension; shape becomes (height, width, channels)
    feature_maps = layer_output[0].numpy() if hasattr(layer_output, "numpy") else layer_output[0]
    
    num_filters = feature_maps.shape[-1]
    
    # Decide on grid size: here we choose 8 columns, and compute number of rows
    cols = 8
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle("Feature Maps from First Conv2D Layer", fontsize=16)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Extract the feature map for the current filter
        feature_map = feature_maps[:, :, i]
        
        # Normalize the feature map for better visualization
        feature_map -= feature_map.mean()
        feature_map /= (feature_map.std() + 1e-5)
        feature_map *= 64
        feature_map += 128
        feature_map = np.clip(feature_map, 0, 255).astype('uint8')
        
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"Filter {i+1}")
        ax.axis('off')
    
    # Turn off any unused subplots
    for j in range(i+1, rows * cols):
        row = j // cols
        col = j % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__=='__main__':
    identifier = "patch_size25_frames_10_ref_test"
    visualize_feature_maps(
        model_path="cnn_model_parameters/my_model.h5",
        data_path="ml_data/"+identifier+"/test_data.npy",
        sample_index=0
    )
