EEG ADHD Detection App
This project is a web-based application developed using Streamlit for detecting ADHD using EEG signals. The application combines EEG signal processing techniques, classical machine learning (Random Forest), and deep learning (a hybrid CNN-LSTM model) for binary classification of ADHD vs. non-ADHD EEG data.

Features
Upload EEG dataset in CSV or Excel format.

Visualize raw and filtered EEG signals using MNE.

Apply bandpass filtering (4–40 Hz).

Remove ocular and muscular artifacts using Independent Component Analysis (ICA).

Compute and compare Power Spectral Density (PSD) before and after filtering.

Train a Random Forest classifier to analyze feature importance.

Build and train a deep learning model combining CNN and LSTM layers.

Display model performance: accuracy, loss curves, confusion matrix, and classification report.

Predict ADHD status on newly uploaded EEG samples.
