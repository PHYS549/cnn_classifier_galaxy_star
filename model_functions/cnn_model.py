import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def identifier_model(data_identifier, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5):
    model_identifier = data_identifier + f"-epochs-{epochs}-batch_size-{batch_size}-{pooling_scheme}-dropout_rate-{dropout_rate}"
    return  model_identifier

def cnn_train_model(data_identifier, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5):
    # -------------------------------
    # Load the Data
    # -------------------------------
    model_identifier = identifier_model(data_identifier, epochs, batch_size, pooling_scheme, dropout_rate)
    X_train = np.load('ml_data/'+data_identifier+'/train_data.npy')
    y_train = np.load('ml_data/'+data_identifier+'/train_targets.npy')

    X_val = np.load('ml_data/'+data_identifier+'/val_data.npy')
    y_val = np.load('ml_data/'+data_identifier+'/val_targets.npy')

    X_test = np.load('ml_data/'+data_identifier+'/test_data.npy')
    y_test = np.load('ml_data/'+data_identifier+'/test_targets.npy')

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
    if pooling_scheme == 'AveragePooling':
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
            AveragePooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            AveragePooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            AveragePooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
    elif pooling_scheme == 'MaxPooling':
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
    else:
        print("Please give either MaxPooling or AveragePooling as your pooling scheme option.")
        return

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # -------------------------------
    # Train the Model
    # -------------------------------
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val)
    )

    # -------------------------------
    # Plot and Save Training and Validation Loss
    # -------------------------------
    # Create results directory if it doesn't exist
    result_dir = "result_plots"
    os.makedirs(result_dir, exist_ok=True)

    # Plot and Save Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the loss curve figure
    loss_plot_filename = f"{result_dir}/{model_identifier}_loss_curve.png"
    plt.savefig(loss_plot_filename)
    print(f"Loss curve saved to {loss_plot_filename}")
    plt.close()

    # Plot and Save Accuracy Curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the accuracy curve figure
    accuracy_plot_filename = f"{result_dir}/{model_identifier}_accuracy_curve.png"
    plt.savefig(accuracy_plot_filename)
    print(f"Accuracy curve saved to {accuracy_plot_filename}")
    plt.close()

    # -------------------------------
    # Evaluate the Model
    # -------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

    # -------------------------------
    # Save the Model
    # -------------------------------
    save_dir = "cnn_model_parameters"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_identifier+"_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")


def visualize_feature_maps(data_identifier, model_identifier, sample_index=None):
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
    model_path="cnn_model_parameters/"+model_identifier+"_model.h5"
    data_path="ml_data/"+data_identifier+"/test_data.npy"
                
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
    if sample_index == None:
        sample_index = np.random.randint(len(X_test))

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
    result_dir = "result_plots"
    plot_filename = f"{result_dir}/{model_identifier}_feature_map.png"
    plt.savefig(plot_filename)
    print(f"Feature Map saved to {plot_filename}")
    plt.close()

def cnn_test_model(data_identifier, model_identifier):
    # -------------------------------
    # Load the Data
    # -------------------------------
    X_test = np.load('ml_data/'+data_identifier+'/test_data.npy')
    y_test = np.load('ml_data/'+data_identifier+'/test_targets.npy')

    # -------------------------------
    # Preprocess the Data
    # -------------------------------
    num_classes = 2
    y_test = to_categorical(y_test, num_classes)

    # -------------------------------
    # Load and Evaluate the Model
    # -------------------------------
    save_dir = "cnn_model_parameters"
    model_path = os.path.join(save_dir, model_identifier+"_model.h5")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    model = load_model(model_path)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 0)
    print('\n')
    print(f"Data Set: {data_identifier}")
    print(f"Model set {model_identifier} accuracy:{test_acc}")

    return test_acc

# Example usage:
if __name__=='__main__':
    identifier = "patch_size25_frames_10_ref_test"
    visualize_feature_maps(
        model_path="cnn_model_parameters/my_model.h5",
        data_path="ml_data/"+identifier+"/test_data.npy",
        sample_index=0
    )
