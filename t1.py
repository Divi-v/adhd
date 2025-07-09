import numpy as np  # Linear algebra
import pandas as pd  # Data processing, CSV file I/O
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, precision_score, recall_score, f1_score
from math import sqrt
import streamlit as st

# Streamlit app title
st.title("EEG ADHD Detection App")

# === Step 1: Upload EEG Dataset ===
uploaded_file = st.file_uploader("Upload EEG dataset/report (CSV format)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Split the data into normal and ADHD datasets
    split_index = min(1514, len(data))  # Ensure the split index is valid
    data_normal = data[:split_index]
    data_adhd = data[split_index:]

    # Debugging: Check the shapes of the datasets
    st.write("Normal Data Shape:", data_normal.shape)
    st.write("ADHD Data Shape:", data_adhd.shape)

    # Get channel names
    channel_names = data.columns.tolist()
    channel_names.remove('class')

    # Define sampling frequency
    sfreq = 128  # Sampling frequency in Hz

    # Create MNE info object
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

    # Set the montage using standard 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Convert DataFrame to NumPy array for ADHD data
    eeg_adhd_data = data_adhd.drop('class', axis=1)

    # Check the shape of the EEG ADHD data
    st.write("EEG ADHD Data Shape:", eeg_adhd_data.shape)  # Check shape

    # Check if the data is empty
    if eeg_adhd_data.shape[0] == 0:
        st.warning("EEG ADHD Data is empty!")
    else:
        eeg_adhd_data = eeg_adhd_data.values.T  # Transpose to have channels as rows and samples as columns
        # Create RawArray object with EEG data and MNE info
        raw_adhd = mne.io.RawArray(eeg_adhd_data, info)
        st.write(raw_adhd)  # Display the Raw object

        # Plot raw EEG data
        mne.set_config('MNE_BROWSE_RAW_SIZE', '16,8')
        fig_raw_adhd = raw_adhd.plot(n_channels=len(channel_names), scalings='auto', title='Raw EEG Data', show=False)
        st.pyplot(fig_raw_adhd)
 
    # Process normal data similarly
    eeg_normal_data = data_normal.drop('class', axis=1)
    eeg_normal_data = eeg_normal_data.values.T  # Transpose to have channels as rows and samples as columns
    raw_normal = mne.io.RawArray(eeg_normal_data, info)
    fig_raw_normal = raw_normal.plot(n_channels=len(channel_names), scalings='auto', title='Raw EEG Data', show=False)
    st.pyplot(fig_raw_normal)

    # Apply bandpass filter (4-40 Hz) only if data is not empty
    if eeg_adhd_data.shape[0] > 0:
        raw_adhd_filtered = raw_adhd.copy().filter(l_freq=4, h_freq=40, method='fir', verbose=False)

    # Apply bandpass filter to normal data
    raw_normal_filtered = raw_normal.copy().filter(l_freq=4, h_freq=40, method='fir', verbose=False)

    # Plot PSD before and after filtering
    fig = plt.figure(figsize=(12, 6))

    # PSD Before Filtering
    ax1 = fig.add_subplot(2, 1, 1)
    raw_normal.plot_psd(fmax=60, ax=ax1, show=False)  # Replace compute_psd with plot_psd
    ax1.set_title('PSD Before Filtering')

    # PSD After Filtering
    ax2 = fig.add_subplot(2, 1, 2)
    raw_normal_filtered.plot_psd(fmax=60, ax=ax2, show=False)  # Replace compute_psd with plot_psd
    ax2.set_title('PSD After Filtering')

    plt.tight_layout()
    st.pyplot(fig)

    # Plot filtered raw data
    fig_filtered_normal = raw_normal_filtered.plot(scalings='auto', show=False)
    st.pyplot(fig_filtered_normal)

    if eeg_adhd_data.shape[0] > 0:
        fig_filtered_adhd = raw_adhd_filtered.plot(scalings='auto', show=False)
        st.pyplot(fig_filtered_adhd)

    # Initialize ICA for normal data
    ica_normal = ICA(n_components=19, random_state=42)
    ica_normal.fit(raw_normal_filtered)

    # Identify components related to ocular artifacts
    eog_inds, scores = ica_normal.find_bads_eog(raw_normal_filtered, ch_name=['Fp1', 'Fp2', 'F7', 'F8'], threshold=3)
    st.write("EOG Indices:", eog_inds)

    # Plot ICA components in topographic maps
    fig_ica_components = ica_normal.plot_components(show=False)
    st.pyplot(fig_ica_components)

    # Plot ICA components in waveform
    fig_ica_sources = ica_normal.plot_sources(raw_normal_filtered, show=False)
    plt.set_cmap('viridis')  # Change 'viridis' to your desired colorscale
    st.pyplot(fig_ica_sources)

    # Exclude identified components from the ICA decomposition
    ica_normal.exclude = [2, 4, 17, 13, 12, 18]

    # Apply ICA to remove ocular artifacts
    cleaned_raw_normal = raw_normal_filtered.copy()
    cleaned_eeg_normal = ica_normal.apply(cleaned_raw_normal)

    # Plot cleaned normal EEG data
    mne.set_config('MNE_BROWSE_RAW_SIZE', '16,8')
    fig_cleaned_normal = cleaned_eeg_normal.plot(scalings='auto', show=False)
    st.pyplot(fig_cleaned_normal)

    # Initialize ICA for ADHD data if it is not empty
    if eeg_adhd_data.shape[0] > 0:
        ica_adhd = ICA(n_components=19, random_state=42)
        ica_adhd.fit(raw_adhd_filtered)

        # Plot ICA components in topographic maps
        fig_ica_components_adhd = ica_adhd.plot_components(show=False)
        st.pyplot(fig_ica_components_adhd)

        # Plot ICA components in waveform
        fig_ica_sources_adhd = ica_adhd.plot_sources(raw_adhd_filtered, show=False)
        plt.set_cmap('viridis')  # Change 'viridis' to your desired colorscale
        st.pyplot(fig_ica_sources_adhd)

        # Exclude identified components from the ICA decomposition
        ica_adhd.exclude = [6, 10, 14, 16]

        # Apply ICA to remove ocular artifacts
        cleaned_raw_adhd = raw_adhd_filtered.copy()
        cleaned_eeg_adhd = ica_adhd.apply(cleaned_raw_adhd)

        # Plot cleaned ADHD EEG data
        fig_cleaned_adhd = cleaned_eeg_adhd.plot(scalings='auto', show=False)
        st.pyplot(fig_cleaned_adhd)

    # Prepare data for machine learning
    eeg_normal = cleaned_eeg_normal.get_data()
    eeg_normal_df = pd.DataFrame(eeg_normal.T, columns=channel_names, index=None)

    # Check if cleaned_eeg_adhd is empty before creating the DataFrame
    if eeg_adhd_data.shape[0] > 0:
        eeg_adhd = cleaned_eeg_adhd.get_data()
        eeg_adhd_df = pd.DataFrame(eeg_adhd.T, columns=channel_names, index=None)
    else:
        # Create an empty DataFrame with the correct columns if no ADHD data is available
        eeg_adhd_df = pd.DataFrame(columns=channel_names)

    # Combine normal and ADHD data
    eeg_df = pd.concat([eeg_normal_df, eeg_adhd_df], axis=0, ignore_index=True)
    eeg_df.reset_index(drop=True, inplace=True)

    # Add class labels
    class_values = data['class'].iloc[:len(eeg_df)]  # Ensure class values match the length of eeg_df
    eeg_df = pd.concat([eeg_df, class_values.reset_index(drop=True)], axis=1)

    # Machine learning with Random Forest
    # Ensure that we sample only if there are enough rows
    sample_size = min(100000, len(eeg_df))
    X = eeg_df.drop(columns=['class']).sample(n=sample_size, random_state=42)  # Assuming 'class' is your target variable
    y = eeg_df['class'].sample(n=sample_size, random_state=42)

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Fit the classifier to your data
    rf_classifier.fit(X, y)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Create a DataFrame to store feature importances along with feature names
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance values
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plotting feature importances
    fig_feature_importance = plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    st.pyplot(fig_feature_importance)

    # Prepare data for deep learning
    X = eeg_df.iloc[:, :-1].values  # Features (excluding the last column)
    y = eeg_df.iloc[:, -1].values   # Labels (last column)

    # Check for NaN values
    st.write("Any NaN in X:", np.any(np.isnan(X)))
    st.write("Any NaN in y:", np.any(np.isnan(y)))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input data to match the expected shape for RNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Check shapes
    st.write("X_train shape:", X_train.shape)  # Should be (num_samples, timesteps, num_features)
    st.write("y_train shape:", y_train.shape)    # Should be (num_samples,)

    # Define model architecture
    input_shape = (X_train.shape[1], 1)

    # Input layer for RNN (LSTM)
    inputs = Input(shape=input_shape)

    # RNN (LSTM) layer
    lstm_output = LSTM(64, return_sequences=True)(inputs)
    lstm_output = LSTM(64)(lstm_output)

    # CNN layers
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv4 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling1D(pool_size=2)(conv4)
    flatten_cnn = Flatten()(pool4)

    # Concatenate outputs from RNN and CNN
    concatenated_outputs = Concatenate()([lstm_output, flatten_cnn])

    # Fully connected layers
    fc_output = Dense(128, activation='relu')(concatenated_outputs)
    dropout_fc = Dropout(0.5)(fc_output)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(dropout_fc)  # Adjust output units and activation function as needed

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Print model summary
    st.write(model.summary())

    # Train the model
    batch_size = 64
    num_epochs = 25


    # Training
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Evaluate on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    st.write(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Plot training and validation accuracy vs. epoch
    fig_accuracy = plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    st.pyplot(fig_accuracy)

    # Predictions and evaluation
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    st.write("Classification Report:")
    st.text(classification_report(y_val, y_pred_binary))

    # Calculate RMSE, precision, recall, and F1 score
    st.write("RMSE:", sqrt(mean_squared_error(y_val, y_pred_binary)))
    st.write("Precision:", precision_score(y_val, y_pred_binary, average='binary'))
    st.write("Recall:", recall_score(y_val, y_pred_binary, average='binary'))
    st.write("F1 Score:", f1_score(y_val, y_pred_binary, average='binary'))

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred_binary)

    # Plot confusion matrix
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig_cm)

    # Plot training and validation loss vs. epoch
    fig_loss = plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    st.pyplot(fig_loss)

    # === Step 2: Predict ADHD Status ===
    st.header("Predict ADHD Status")
    st.write("Upload new EEG data for prediction.")

    # File uploader for new EEG data
    new_data_file = st.file_uploader("Upload new EEG data (CSV format)", type=["csv"], key="new_data")
    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)

        # Ensure the new data has the same columns as the training data
        if set(channel_names).issubset(new_data.columns):
            new_eeg_data = new_data[channel_names].values.T  # Transpose to have channels as rows and samples as columns
            new_raw = mne.io.RawArray(new_eeg_data, info)

            # Apply the same preprocessing steps
            new_raw_filtered = new_raw.copy().filter(l_freq=4, h_freq=40, method='fir', verbose=False)

            # Prepare the data for prediction
            new_eeg_data = new_raw_filtered.get_data()
            new_eeg_data = new_eeg_data.reshape(new_eeg_data.shape[1], new_eeg_data.shape[0], 1)  # Reshape for RNN

            # Make predictions
            predictions = model.predict(new_eeg_data)
            predictions_binary = (predictions > 0.5).astype(int)

            # Display predictions
            st.write("Predictions (0 = Normal, 1 = ADHD):")
            st.write(predictions_binary.flatten())

            if predictions_binary == 0:
                st.write("No ADHD Detected")

            else:
                st.write("ADHD Detected")
        else:
            st.warning("The uploaded data does not contain the required EEG channels.")
else:
    st.warning("Please upload a CSV file to proceed.")