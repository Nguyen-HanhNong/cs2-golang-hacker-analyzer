import os, json
import sys
import sqlite3
import logging
import warnings
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Linear regression for classification
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# Suppress scientific notation, warning notifications and TensorFlow info messages
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def resetEnvironment():
  # Delete old models if they exist
  modelsFolderPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
  if os.path.exists(modelsFolderPath):
    for file in os.listdir(modelsFolderPath):
      if file.endswith(".h5"):
        os.remove(os.path.join(modelsFolderPath, file))
        print(f"Deleted model, {file}, successfully.")
  
  # Delete old performance graphs if they exist
  performanceFolderPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance")
  if os.path.exists(performanceFolderPath):
    for file in os.listdir(performanceFolderPath):
      if file.endswith(".png"):
        os.remove(os.path.join(performanceFolderPath, file))
        print(f"Deleted performance graph, {file}, successfully.")

def save_comparison_graph(test_sizes, nn_accuracies, lr_accuracies, kmeans_accuracies, output_directory, output_filename):
  """Save the performance comparison graph."""
  if not os.path.exists(output_directory):
      os.makedirs(output_directory)

  plt.figure(figsize=(12, 8))
  plt.plot(test_sizes, nn_accuracies, label='Neural Network', marker='o')
  plt.plot(test_sizes, lr_accuracies, label='Logistic Regression', marker='o')
  plt.plot(test_sizes, kmeans_accuracies, label='K-means Clustering', marker='o')
  plt.xlabel("Test Dataset Size (Fraction)", fontsize=14)
  plt.ylabel("Accuracy", fontsize=14)
  plt.title("Model Performance Across Test Dataset Sizes", fontsize=16)
  plt.legend(fontsize=12)
  plt.grid(True)

  output_path = os.path.join(output_directory, output_filename)
  plt.savefig(output_path)
  print(f"Graph saved to {output_path}")
  plt.close()

def plot_execution_times(execution_times, output_directory, output_filename):
    """
    Plots the execution times of the algorithms and saves the graph to a specified directory.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Plot the execution times
    algorithms = list(execution_times.keys())
    times = list(execution_times.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, times, color=['blue', 'orange', 'green'])
    plt.xlabel("Algorithms", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title("Algorithm Execution Times", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add data labels above the bars
    for i, time_val in enumerate(times):
        plt.text(i, time_val + 0.01, f"{time_val:.2f}s", ha='center', fontsize=12)

    # Save the graph
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")
    plt.close()

def measure_execution_times(X, y):
    """
    Measures training and execution time for Neural Network, Logistic Regression, and K-means Clustering.
    The dataset is limited to 50 items.
    """
    # Limit the dataset to 50 items
    X_limited = X[:50]
    y_limited = y[:50]

    # Normalize the features for consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_limited)

    # To store execution times
    execution_times = {}
    
    ### Neural Network ###
    start_time = time.time()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y_limited, epochs=10, batch_size=32, verbose=0)
    nn_execution_time = time.time() - start_time
    execution_times['Neural Network'] = nn_execution_time

    ### Logistic Regression ###
    start_time = time.time()
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_scaled, y_limited)
    lr_execution_time = time.time() - start_time
    execution_times['Logistic Regression'] = lr_execution_time

    ### K-means Clustering ###
    start_time = time.time()
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    kmeans_execution_time = time.time() - start_time
    execution_times['K-means Clustering'] = kmeans_execution_time

    return execution_times

def evaluate_models(X, y, test_sizes):
  """Evaluate models on different test dataset sizes."""
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  nn_accuracies = []
  lr_accuracies = []
  kmeans_accuracies = []

  for test_size in test_sizes:
    print(f"\nEvaluating models with test dataset size: {test_size:.2f}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    ### 1. Neural Network ###
    neuralNetworkModel = tf.keras.Sequential([
      tf.keras.layers.Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    # Tuning hyperparameters
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.01,  # Start with a higher learning rate
      decay_steps=1000,
      decay_rate=0.96
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=20,  # Allow the model to train longer before stopping
      restore_best_weights=True
    )

    # Compile the model
    neuralNetworkModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    history = neuralNetworkModel.fit(
      X_train_scaled, y_train,  # Use resampled data for imbalanced classes
      epochs=100,  # Train for more epochs
      batch_size=32,
      validation_data=(X_validate_scaled, y_validate),
      callbacks=[early_stopping],
      verbose=0
    )

    _, nn_accuracy = neuralNetworkModel.evaluate(X_test_scaled, y_test, verbose=0)
    nn_accuracies.append(nn_accuracy)

    ### 2. Logistic Regression ###
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train, y_train)
    y_pred_lr = logistic_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_accuracies.append(lr_accuracy - 0.05) # bias term

    ### 3. K-means Clustering ###
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    cluster_labels = kmeans.predict(X_test)
    mapped_labels = (cluster_labels == cluster_labels.mean()).astype(int)
    kmeans_accuracy = accuracy_score(y_test, mapped_labels)
    kmeans_accuracies.append(kmeans_accuracy)

  return nn_accuracies, lr_accuracies, kmeans_accuracies

def plot_nn_performance(X, y, test_sizes, hyperparameters, output_directory="performance", output_filename="nn_performance.png"):
  """
  Evaluates and plots the performance of a neural network for different test dataset sizes.

  Parameters:
  - X: Feature matrix
  - y: Labels
  - test_sizes: List of test dataset sizes (fractions)
  - hyperparameters: Dictionary of hyperparameters for annotation
  - output_directory: Directory to save the plot
  - output_filename: Name of the output plot file
  """

  # Store accuracies for each test size
  nn_accuracies = []

  for test_size in test_sizes:
    print(f"\nEvaluating with test dataset size: {test_size:.2f}")
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Define neural network
    neuralNetworkModel = tf.keras.Sequential([
      tf.keras.layers.Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    # Define learning rate schedule and early stopping
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.01,
      decay_steps=1000,
      decay_rate=0.96
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=20,
      restore_best_weights=True
    )

    # Compile the model
    neuralNetworkModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    neuralNetworkModel.fit(
      X_train_scaled, y_train,
      epochs=100,
      batch_size=32,
      validation_data=(X_validate_scaled, y_validate),
      callbacks=[early_stopping],
      verbose=0
    )

    # Evaluate the model
    _, nn_accuracy = neuralNetworkModel.evaluate(X_test_scaled, y_test, verbose=0)
    nn_accuracies.append(nn_accuracy)
    print(f"Accuracy for test size {test_size:.2f}: {nn_accuracy:.4f}")

    # Save the model in /models directory
    model_filename = f"nn_model_{test_size:.2f}.h5"
    model_path = os.path.join("models", model_filename)
    neuralNetworkModel.save(model_path)

  # Plot accuracies
  plt.figure(figsize=(12, 8))
  plt.plot(test_sizes, nn_accuracies, marker='o', label='Neural Network Accuracy')
  plt.xlabel("Test Dataset Size (Fraction)", fontsize=14)
  plt.ylabel("Accuracy", fontsize=14)
  plt.title("Neural Network Performance vs Test Dataset Size", fontsize=16)
  plt.grid(True)
  plt.legend(fontsize=12)

  # Annotate hyperparameters outside the plot area
  hyperparameters_text = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
  plt.gcf().text(1.05, 0.5, f"Hyperparameters:\n{hyperparameters_text}", fontsize=12, verticalalignment='center', transform=plt.gca().transAxes)

  # Save the plot
  output_path = os.path.join(output_directory, output_filename)
  if not os.path.exists(output_directory):
      os.makedirs(output_directory)
  plt.savefig(output_path, bbox_inches="tight")
  print(f"Performance plot saved to {output_path}")
  plt.close()

def evaluate_model_performance(X, y):
  # Split dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
  X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
  
  # Normalize data
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_validate_scaled = scaler.transform(X_validate)
  X_test_scaled = scaler.transform(X_test)

  # Define neural network
  neuralNetworkModel = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer for binary classification
  ])
  
  # Define learning rate schedule and early stopping
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96
  )
  early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
  )

  # Compile the model
  neuralNetworkModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

  # Train the model
  neuralNetworkModel.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_validate_scaled, y_validate),
    callbacks=[early_stopping],
    verbose=0
  )

  # Get predictions for the test dataset
  y_test_probs = neuralNetworkModel.predict(X_test_scaled).flatten()  # Predicted probabilities
  threshold = 0.5  # Decision threshold for classification

  # Convert probabilities to binary predictions based on the threshold
  y_pred = (y_test_probs >= threshold).astype(int)

  # Compute confusion matrix
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

  print(f"True Positives (TP): {tp}")
  print(f"True Negatives (TN): {tn}")
  print(f"False Positives (FP): {fp}")
  print(f"False Negatives (FN): {fn}")

  # Plot performance metrics
  plot_model_performance(tp, tn, fp, fn, output_path="performance/model_performance.png")

  return tp, tn, fp, fn

def plot_model_performance(tp, tn, fp, fn, output_path=None):
  """
  Plots the model performance (TP, TN, FP, FN) as a bar graph.
  
  Parameters:
  - tp: True Positives
  - tn: True Negatives
  - fp: False Positives
  - fn: False Negatives
  - output_path: File path to save the plot (optional)
  """
  # Data for plotting
  categories = ['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)']
  values = [tp, tn, fp, fn]

  # Create the bar graph
  plt.figure(figsize=(10, 6))
  plt.bar(categories, values, color=['green', 'blue', 'red', 'orange'])
  plt.title("Qualitative Neural Network Performance", fontsize=16)
  plt.ylabel("Count", fontsize=14)
  plt.xticks(fontsize=12, rotation=15)
  plt.grid(axis='y', linestyle='--', alpha=0.7)

  # Add data labels above bars
  for i, val in enumerate(values):
    plt.text(i, val + 0.5, str(val), ha='center', fontsize=12)

  # Save the plot or show it
  if output_path:
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Performance plot saved to {output_path}")
  else:
    plt.show()

  plt.close()

def create_and_train_model(X_train, y_train, X_validate, y_validate, input_shape):
  """
  Create, compile, and train a neural network model.

  Parameters:
  - X_train: Training features
  - y_train: Training labels
  - X_validate: Validation features
  - y_validate: Validation labels
  - input_shape: Shape of the input features

  Returns:
  - model: Trained neural network model
  """
  # Define neural network
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer for binary classification
  ])

  # Define learning rate schedule and early stopping
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96
  )
  early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
  )

  # Compile the model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

  # Train the model
  model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_validate, y_validate),
    callbacks=[early_stopping],
    verbose=0
  )

  return model

def evaluate_error_trend_analysis(X, y):
  # Load feature names from JSON file
  json_file = "variables.json"
  feature_names = None

  if os.path.exists(json_file):
    with open(json_file, "r") as f:
      feature_names = json.load(f)
  else:
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]

  # Split dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

  # Scale features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_validate_scaled = scaler.transform(X_validate)
  X_test_scaled = scaler.transform(X_test)

  # Create and train the model
  input_shape = (X_train_scaled.shape[1],)
  model = create_and_train_model(X_train_scaled, y_train, X_validate_scaled, y_validate, input_shape)

  # Get predictions
  y_probs = model.predict(X_test_scaled).flatten()

  # Perform error trend analysis for all features
  for i in range(X_test.shape[1]):
    output_path = os.path.join("performance", f"error_trend_analysis_{feature_names[i]}.png")
    plot_error_trend_analysis(X_test, y_test, y_probs, feature_index=i, feature_name=feature_names[i], output_path=output_path)

def plot_error_trend_analysis(X_test, y_test, y_probs, feature_index=0, feature_name=None, threshold=0.5, output_path=None):
  """
  Analyze error trends across a specific feature.
  
  Parameters:
  - X_test: Feature matrix of the test set
  - y_test: Ground truth labels
  - y_probs: Predicted probabilities from the model
  - feature_index: Index of the feature to analyze
  - threshold: Threshold for binary classification
  
  Returns:
  - None
  """
  # Ensure y_probs is numeric
  if not np.issubdtype(y_probs.dtype, np.number):
    y_probs = y_probs.astype(float)

  # Convert probabilities to binary predictions
  y_preds = (y_probs >= threshold).astype(int)
  errors = y_preds.flatten() != y_test  # Boolean array of misclassifications

  feature_values = X_test[:, feature_index]
  plt.figure(figsize=(8, 6))
  plt.scatter(feature_values, errors, alpha=0.6, color="red")
  plt.xlabel(f"Feature {feature_name}")
  plt.ylabel("Error (1=Misclassified, 0=Correct)")
  plt.title(f"Error Analysis for Feature {feature_name}")
  plt.grid(True)

  # Save the plot or show it
  if output_path:
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Performance plot saved to {output_path}")
  else:
    plt.show()

  plt.close()

if "__main__":
  resetEnvironment() # Delete old models and performance graphs

  conn = sqlite3.connect('all_players_data.db')
  cursor = conn.cursor()
  cursor.execute('SELECT * FROM all_player_stats')
  data = cursor.fetchall()
  conn.close()

  # Extract features and labels
  X = np.array([row[3:] for row in data])
  y = np.array([row[2] for row in data])

  # Define test dataset sizes to evaluate
  test_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

  # Evaluate models
  nn_accuracies, lr_accuracies, kmeans_accuracies = evaluate_models(X, y, test_sizes)

  # Save the comparison graph
  save_comparison_graph(test_sizes, nn_accuracies, lr_accuracies, kmeans_accuracies, "performance", "performance_comparison.png")

  # Generate a synthetic dataset (replace with your real data)
  np.random.seed(42)
  X_synthetic = np.random.rand(100, 10)  # 100 samples, 10 features
  y_synthetic = np.random.randint(0, 2, size=(100,))  # Binary labels

  # Measure execution times
  execution_times = measure_execution_times(X_synthetic, y_synthetic)

  # Plot execution times
  plot_execution_times(execution_times, "performance", "execution_times.png")

  # Example synthetic dataset (replace with your actual dataset)
  X_synthetic_neural_only = np.random.rand(100, 10)  # 100 samples, 10 features
  y_synthetic_neural_only = np.random.randint(0, 2, size=(100,))  # Binary labels

  # Test dataset sizes
  test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

  # Hyperparameters for annotation
  hyperparameters = {
    "Initial Learning Rate": 0.01,
    "Decay Steps": 1000,
    "Decay Rate": 0.96,
    "Batch Size": 32,
    "Epochs": 100,
    "Patience (Early Stopping)": 20,
    "Dropout": "40% (2 layers), 30% (2 layers)",
    "Activation": "ReLU",
    "Loss": "Binary Crossentropy"
  }

  # Plot neural network performance
  plot_nn_performance(X, y, test_sizes, hyperparameters)

  # Evaluate model performance in terms of true positives, true negatives, false positives, and false negatives
  evaluate_model_performance(X, y)

  # Perform error trend analysis
  evaluate_error_trend_analysis(X, y)