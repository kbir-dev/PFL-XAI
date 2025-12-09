import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd
from xai_utils import (
    load_model, get_gradcam_explanation, get_integrated_gradients_explanation,
    plot_time_series_attribution, get_activity_name, load_test_sample,
    ACTIVITY_LABELS, IDX_TO_ACTIVITY
)
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="PFL-XAI : Personalized Federated Learning with Explainable AI for Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# Define sensor column names for better interpretability
SENSOR_COLUMNS = []
locations = ['hand', 'chest', 'ankle']
sensor_types = [
    'temp',
    'acc16_x', 'acc16_y', 'acc16_z',
    'acc6_x', 'acc6_y', 'acc6_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'orientation_1', 'orientation_2', 'orientation_3', 'orientation_4'
]

for location in locations:
    for sensor in sensor_types:
        SENSOR_COLUMNS.append(f"{location}_{sensor}")

# CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #4169E1;
    text-align: center;
}
.sub-header {
    font-size: 1.8rem;
    color: #1E90FF;
}
.info-box {
    background-color: #F0F8FF;
    padding: 20px;
    border-radius: 5px;
    margin-bottom: 20px;
}
.model-metrics {
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>PFL-XAI : Personalized Federated Learning with Explainable AI <br>for Human Activity Recognition</h1>", unsafe_allow_html=True)

# Sidebar for model selection and configuration
st.sidebar.markdown("## Model Configuration")

# Function to list all available models
@st.cache_data
def get_available_models():
    # Try different possible model directory paths
    possible_paths = [
        "../fedbn/models",         # Up one level
        "fedbn/models",           # Same level
        "../../fedbn/models",     # Up two levels
        "../models",              # Up one level, directly to models
        "models"                  # Same level, directly to models
    ]
    
    models = []
    model_dir = None
    
    # Find the first valid path
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            model_dir = path
            st.sidebar.success(f"Found model directory: {path}")
            break
    
    if model_dir is None:
        st.sidebar.error("Could not find models directory")
        return []
    
    # List the models
    try:
        for filename in os.listdir(model_dir):
            if filename.endswith('.h5') and filename.startswith('fedbn_model_round_'):
                try:
                    round_num = int(filename.split('_')[-1].split('.')[0])
                    models.append((round_num, os.path.join(model_dir, filename)))
                except ValueError:
                    continue
        return sorted(models)
    except Exception as e:
        st.sidebar.error(f"Error listing models: {e}")
        return []

# Get available models
available_models = get_available_models()

# Default model path and round
model_path = None
selected_round = None

if available_models:
    try:
        model_rounds = [model[0] for model in available_models]
        selected_round = st.sidebar.selectbox("Select model round", model_rounds, index=len(model_rounds)-1)
        model_path = [model[1] for model in available_models if model[0] == selected_round][0]
    except Exception as e:
        st.error(f"Error selecting model: {e}")
        st.stop()
else:
    st.error("No model files found. Please check your directory structure.")
    st.stop()

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_cached_model(model_path):
    try:
        # Set TF to allow growth to avoid memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile the model
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Print model summary for verification
        print("Model input shape:", model.input_shape)

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cached_model(model_path)
if model is None:
    st.stop()

# Display model info in sidebar
st.sidebar.success(f"Loaded model from round {selected_round}")
st.sidebar.info("Model Architecture: CNN with BatchNormalization")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Single Sample Analysis", "Client Data Analysis", "Dataset Viewer"])
#tab3 = "Round Comparison"

with tab1:
    st.markdown("<h2 class='sub-header'>Single Sample Explainability</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Upload a single time-series sample to analyze and explain the model's prediction.
    The sample should be in .npz format with an 'X' array of shape (timesteps, features).
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader for single sample
    uploaded_file = st.file_uploader("Upload a .npz file with X array", type="npz", key="single_sample")
    
    # Sample selection from existing data
    st.markdown("### OR Select from existing samples")
    col1, col2 = st.columns(2)
    with col1:
        client_id = st.selectbox("Select client", list(range(20)))
    with col2:
        sample_source = st.radio("Data split", ["test", "train"])
    
    sample_idx = st.number_input("Sample index", min_value=0, value=0)
    
    load_sample = st.button("Load Sample")
    
    # Sample data container
    X_sample = None
    y_true = None
    
    if uploaded_file is not None:
        data = np.load(uploaded_file)
        try:
            X_sample = data['X']
            st.success(f"Loaded sample with shape: {X_sample.shape}")
            if 'y' in data:
                y_true = int(data['y'])
                activity_id = IDX_TO_ACTIVITY.get(y_true, y_true)
                st.info(f"Ground truth label: {y_true} ({ACTIVITY_LABELS.get(activity_id, 'Unknown')})")
        except:
            st.error("Invalid data format. Expected 'X' array in the npz file.")
    
    elif load_sample:
        try:
            # Try different possible paths for client data
            possible_paths = [
                f"../clients_data/client_{client_id}_{sample_source}.npz",  # Up one level
                f"clients_data/client_{client_id}_{sample_source}.npz",      # Same level
                f"../../clients_data/client_{client_id}_{sample_source}.npz"  # Up two levels
            ]
            
            sample_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    sample_path = path
                    break
                    
            if sample_path is None:
                st.error(f"Could not find client data file for client {client_id}")
                st.stop()
                
            data = np.load(sample_path)
            X_all = data['X']
            y_all = data['y']
            
            if sample_idx >= len(X_all):
                st.error(f"Sample index out of range. Max index: {len(X_all)-1}")
            else:
                X_sample = X_all[sample_idx]
                y_true = int(y_all[sample_idx])
                activity_id = IDX_TO_ACTIVITY.get(y_true, y_true)
                st.success(f"Loaded sample {sample_idx} from client {client_id} ({sample_source})")
                st.info(f"Ground truth label: {y_true} ({ACTIVITY_LABELS.get(activity_id, 'Unknown')})")
        except Exception as e:
            st.error(f"Error loading sample: {e}")
    
    # Make prediction if sample is loaded
    if X_sample is not None:
        # Ensure X_sample has the right shape
        if len(X_sample.shape) == 2:  # (timesteps, features)
            X_input = X_sample[np.newaxis, ...]  # Add batch dimension
        else:
            st.error(f"Unexpected input shape: {X_sample.shape}. Expected (timesteps, features)")
            X_input = None
            
        if X_input is not None:
            # Make prediction
            prediction = model.predict(X_input, verbose=0)
            pred_class = np.argmax(prediction, axis=1)[0]
            pred_prob = prediction[0, pred_class]
            
            activity_id = IDX_TO_ACTIVITY.get(pred_class, pred_class)
            activity_name = ACTIVITY_LABELS.get(activity_id, f"Activity {activity_id}")
            
            # Show prediction
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='model-metrics'>Predicted Class: {pred_class} ({activity_name})</div>", 
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='model-metrics'>Confidence: {pred_prob:.4f}</div>", 
                            unsafe_allow_html=True)
            
            # # Show time-series plot
            # st.markdown("### Input Time Series")
            
            # # Create an interactive time-series plot with plotly
            # fig = go.Figure()
            
            # # If we have many features, select a subset for visualization
            # num_features = X_sample.shape[1]
            
            # # Add traces for each feature (or a subset if there are many)
            # max_traces = 6  # Maximum number of traces to show by default
            # for i in range(min(num_features, max_traces)):
            #     feature_name = SENSOR_COLUMNS[i] if i < len(SENSOR_COLUMNS) else f"Feature {i}"
            #     fig.add_trace(go.Scatter(
            #         y=X_sample[:, i],
            #         mode='lines',
            #         name=feature_name
            #     ))
            
            # fig.update_layout(
            #     title='Input Time Series Data',
            #     xaxis_title='Time Step',
            #     yaxis_title='Sensor Value',
            #     height=400
            # )
            # st.plotly_chart(fig, use_container_width=True)
            
            # # XAI methods selection
            # st.markdown("### Generate Explanations")
            # xai_method = st.selectbox(
            #     "Select XAI method",
            #     ["GradCAM", "Integrated Gradients"]
            # )
            
            # if st.button("Generate Explanation"):
            #     st.info(f"Generating {xai_method} explanation...")
                
            #     try:
            #         if xai_method == "GradCAM":
            #             explanation, _ = get_gradcam_explanation(model, X_input, pred_class)
                        
            #             # Display the heatmap
            #             fig, ax = plt.subplots(figsize=(10, 6))
            #             im = ax.imshow(explanation, aspect='auto')
            #             ax.set_title(f"GradCAM Heatmap for Class: {activity_name}")
            #             ax.set_xlabel("Features")
            #             ax.set_ylabel("Time Steps")
            #             plt.colorbar(im, ax=ax)
            #             st.pyplot(fig)
                        
            #             st.markdown("""
            #             **GradCAM Interpretation**: 
            #             The heatmap shows which regions of the time-series data most influenced the model's prediction.
            #             Brighter areas (yellow/red) had a stronger positive contribution to the final classification.
            #             """)
                    
            #         elif xai_method == "Integrated Gradients":
            #             explanation, _ = get_integrated_gradients_explanation(model, X_input, pred_class)
                        
            #             # Display the attribution plot
            #             fig, ax = plt.subplots(figsize=(10, 6))
            #             im = ax.imshow(explanation[0], aspect='auto')
            #             ax.set_title(f"Integrated Gradients Attribution for Class: {activity_name}")
            #             ax.set_xlabel("Features")
            #             ax.set_ylabel("Time Steps")
            #             plt.colorbar(im, ax=ax)
            #             st.pyplot(fig)
                    
            #             # Interactive attribution plot
            #             st.markdown("### Interactive Feature Attribution")
            #             try:
            #                 attribution_fig = plot_time_series_attribution(
            #                     X_sample, 
            #                     explanation[0], 
            #                     feature_names=SENSOR_COLUMNS[:X_sample.shape[1]] if len(SENSOR_COLUMNS) >= X_sample.shape[1] else None
            #                 )
            #                 st.plotly_chart(attribution_fig, use_container_width=True)
            #             except Exception as e:
            #                 st.error(f"Error creating interactive attribution plot: {e}")
                        
            #             st.markdown("""
            #             **Integrated Gradients Interpretation**:
            #             This visualization shows which specific feature values contributed most to the prediction.
            #             Red areas indicate positive contribution, blue areas indicate negative contribution to the predicted class.
            #             """)
            #     except Exception as e:
            #         st.error(f"Error generating explanation: {str(e)}")
            #         st.error(f"Error details: {str(e)}")

with tab2:
    st.markdown("<h2 class='sub-header'>Client Data Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Analyze model performance and explanations on specific client data.
    </div>
    """, unsafe_allow_html=True)
    
    # Client selection
    client_id = st.selectbox("Select client", list(range(20)), key="client_tab")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.slider("Number of samples to analyze", 1, 10, 3)
    with col2:
        random_samples = st.checkbox("Use random samples", value=True)
    
    if st.button("Analyze Client Data", key="analyze_client_data_btn"):
        try:
            # Try different possible paths for client data
            possible_paths = [
                f"../clients_data/client_{client_id}_test.npz",  # Up one level
                f"clients_data/client_{client_id}_test.npz",      # Same level
                f"../../clients_data/client_{client_id}_test.npz"  # Up two levels
            ]
            
            client_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    client_path = path
                    break
                    
            if client_path is None:
                st.error(f"Could not find test data file for client {client_id}")
                st.stop()
                
            data = np.load(client_path)
            X = data['X']
            y = data['y']
            
            # Select samples
            if random_samples:
                sample_indices = np.random.choice(len(X), size=num_samples, replace=False)
            else:
                sample_indices = np.arange(min(num_samples, len(X)))
            
            X_selected = X[sample_indices]
            y_selected = y[sample_indices]
            
            # Make predictions
            predictions = model.predict(X_selected)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Calculate accuracy on these samples
            accuracy = np.mean(pred_classes == y_selected)
            st.success(f"Model accuracy on selected samples: {accuracy:.2f}")
            
            # Show results for each sample
            for i in range(len(X_selected)):
                st.markdown(f"### Sample {i+1}")
                
                pred_class = pred_classes[i]
                true_class = y_selected[i]
                
                pred_activity_id = IDX_TO_ACTIVITY.get(pred_class, pred_class)
                true_activity_id = IDX_TO_ACTIVITY.get(true_class, true_class)
                
                pred_activity_name = ACTIVITY_LABELS.get(pred_activity_id, f"Activity {pred_activity_id}")
                true_activity_name = ACTIVITY_LABELS.get(true_activity_id, f"Activity {true_activity_id}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"True class: {true_class} ({true_activity_name})")
                with col2:
                    if pred_class == true_class:
                        st.success(f"Predicted class: {pred_class} ({pred_activity_name})")
                    else:
                        st.error(f"Predicted class: {pred_class} ({pred_activity_name})")
                
                # try:
                #     # Generate and display explanation
                #     explanation, _ = get_gradcam_explanation(model, X_selected[i:i+1], pred_class)
                    
                #     # Use dynamic feature labels
                #     num_features = X_selected[i].shape[1]
                #     if len(SENSOR_COLUMNS) != num_features:
                #         st.warning(f"SENSOR_COLUMNS length ({len(SENSOR_COLUMNS)}) doesn't match data features ({num_features}). Using generic column names.")
                #         feature_names = [f"feature_{i}" for i in range(num_features)]
                #     else:
                #         feature_names = SENSOR_COLUMNS
                    
                #     # Create a plotly heatmap for better visualization
                #     fig = px.imshow(explanation.T,
                #                    labels=dict(x="Time Step", y="Sensor", color="Attribution"),
                #                    x=list(range(explanation.shape[0])),
                #                    y=feature_names,
                #                    aspect="auto",
                #                    color_continuous_scale="RdBu_r",
                #                    title=f"GradCAM for {pred_activity_name}")
                #     st.plotly_chart(fig, use_container_width=True)
                    
                #     # Show time series with attribution overlay
                #     st.subheader("Time Series with Attribution Overlay")
                #     time_series_fig = plot_time_series_attribution(X_selected[i], explanation, feature_names)
                #     st.plotly_chart(time_series_fig, use_container_width=True)
                # except Exception as e:
                #     st.error(f"Error generating explanation: {e}")
                #     st.write("Error details:", str(e))
                
                # # Draw horizontal line to separate samples
                # if i < len(X_selected) - 1:
                #     st.markdown("---")
        
        except Exception as e:
            st.error(f"Error analyzing client data: {e}")

# with tab3:
#     st.markdown("<h2 class='sub-header'>Round Comparison</h2>", unsafe_allow_html=True)
    
#     st.markdown("""
#     <div class='info-box'>
#     Compare model predictions and explanations across different training rounds.
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         round1 = st.selectbox("Select first round", model_rounds, index=0)
#     with col2:
#         round2 = st.selectbox("Select second round", model_rounds, index=len(model_rounds)-1)
    
#     # Sample selection for comparison
#     st.markdown("### Select sample for comparison")
#     client_comp = st.selectbox("Select client", list(range(20)), key="compare_client")
#     sample_idx_comp = st.number_input("Sample index", min_value=0, value=0, key="compare_sample")
    
#     if st.button("Compare Rounds", key="compare_rounds_btn"):
#         try:
#             # Try different possible paths for client data
#             possible_paths = [
#                 f"../clients_data/client_{client_comp}_test.npz",  # Up one level
#                 f"clients_data/client_{client_comp}_test.npz",      # Same level
#                 f"../../clients_data/client_{client_comp}_test.npz"  # Up two levels
#             ]
            
#             client_path = None
#             for path in possible_paths:
#                 if os.path.exists(path):
#                     client_path = path
#                     break
                    
#             if client_path is None:
#                 st.error(f"Could not find test data file for client {client_comp}")
#                 st.stop()
                
#             data = np.load(client_path)
#             X_all = data['X']
#             y_all = data['y']
            
#             if sample_idx_comp >= len(X_all):
#                 st.error(f"Sample index out of range. Max index: {len(X_all)-1}")
#             else:
#                 X_sample = X_all[sample_idx_comp:sample_idx_comp+1]  # Keep batch dimension
#                 y_true = int(y_all[sample_idx_comp])
                
#                 # Get model paths
#                 model_path1 = [path for r, path in available_models if r == round1][0]
#                 model_path2 = [path for r, path in available_models if r == round2][0]
                
#                 # Load models
#                 model1 = load_cached_model(model_path1)
#                 model2 = load_cached_model(model_path2)
                
#                 if model1 is None or model2 is None:
#                     st.error("Failed to load models for comparison")
#                 else:
#                     # Make predictions
#                     pred1 = model1.predict(X_sample, verbose=0)
#                     pred2 = model2.predict(X_sample, verbose=0)
                    
#                     pred_class1 = np.argmax(pred1, axis=1)[0]
#                     pred_class2 = np.argmax(pred2, axis=1)[0]
                    
#                     activity_id1 = IDX_TO_ACTIVITY.get(pred_class1, pred_class1)
#                     activity_id2 = IDX_TO_ACTIVITY.get(pred_class2, pred_class2)
                    
#                     true_activity_id = IDX_TO_ACTIVITY.get(y_true, y_true)
                    
#                     # Show predictions
#                     st.markdown(f"**Ground Truth: {y_true} ({ACTIVITY_LABELS.get(true_activity_id, 'Unknown')})**")
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.markdown(f"**Round {round1} Prediction**")
#                         if pred_class1 == y_true:
#                             st.success(f"Predicted: {pred_class1} ({ACTIVITY_LABELS.get(activity_id1, 'Unknown')})")
#                         else:
#                             st.error(f"Predicted: {pred_class1} ({ACTIVITY_LABELS.get(activity_id1, 'Unknown')})")
#                         st.info(f"Confidence: {pred1[0, pred_class1]:.4f}")
#                     with col2:
#                         st.markdown(f"**Round {round2} Prediction**")
#                         if pred_class2 == y_true:
#                             st.success(f"Predicted: {pred_class2} ({ACTIVITY_LABELS.get(activity_id2, 'Unknown')})")
#                         else:
#                             st.error(f"Predicted: {pred_class2} ({ACTIVITY_LABELS.get(activity_id2, 'Unknown')})")
#                         st.info(f"Confidence: {pred2[0, pred_class2]:.4f}")
                    
#                     # Generate explanations
#                     st.markdown("### Explanations Comparison")
#                     try:
#                         # Generate explanations without dummy initialization
#                         explanation1, _ = get_gradcam_explanation(model1, X_sample, pred_class1)
#                         explanation2, _ = get_gradcam_explanation(model2, X_sample, pred_class2)
                        
#                         # Use dynamic feature labels
#                         num_features = X_sample[0].shape[1]
#                         if len(SENSOR_COLUMNS) != num_features:
#                             st.warning(f"SENSOR_COLUMNS length ({len(SENSOR_COLUMNS)}) doesn't match data features ({num_features}). Using generic column names.")
#                             feature_names = [f"feature_{i}" for i in range(num_features)]
#                         else:
#                             feature_names = SENSOR_COLUMNS
                        
#                         # Compare explanations using plotly for better visualization
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             fig1 = px.imshow(explanation1.T,
#                                            labels=dict(x="Time Step", y="Sensor", color="Attribution"),
#                                            x=list(range(explanation1.shape[0])),
#                                            y=feature_names,
#                                            aspect="auto",
#                                            color_continuous_scale="RdBu_r",
#                                            title=f"Round {round1} GradCAM")
#                             st.plotly_chart(fig1, use_container_width=True)
                        
#                         with col2:
#                             fig2 = px.imshow(explanation2.T,
#                                            labels=dict(x="Time Step", y="Sensor", color="Attribution"),
#                                            x=list(range(explanation2.shape[0])),
#                                            y=feature_names,
#                                            aspect="auto",
#                                            color_continuous_scale="RdBu_r",
#                                            title=f"Round {round2} GradCAM")
#                             st.plotly_chart(fig2, use_container_width=True)
                        
#                         # Show time series with attribution overlay for both rounds
#                         st.subheader(f"Round {round1} - Time Series with Attribution Overlay")
#                         time_series_fig1 = plot_time_series_attribution(X_sample[0], explanation1, feature_names)
#                         st.plotly_chart(time_series_fig1, use_container_width=True)
#                         st.info("Visualization showing which input features influenced the model's decision")
                        
#                         # Second round visualization
#                         st.subheader(f"Round {round2} - Time Series with Attribution Overlay")
#                         time_series_fig2 = plot_time_series_attribution(X_sample[0], explanation2, feature_names)
#                         st.plotly_chart(time_series_fig2, use_container_width=True)
#                         st.info("Visualization showing which input features influenced the model's decision")
#                     except Exception as e:
#                         st.error(f"Error generating explanations: {str(e)}")
#                         st.write("Error details:", str(e))
                    
#                     # Interpretation
#                     st.markdown("""
#                     **Interpretation**:
#                     Compare the two GradCAM visualizations to see how the model's focus changed between rounds.
#                     Areas with brighter colors indicate features that strongly influenced the model's decision.
#                     """)
                    
#                     # Calculate explanation difference
#                     diff = explanation2 - explanation1
                    
#                     fig, ax = plt.subplots(figsize=(10, 5))
#                     im = ax.imshow(diff, cmap='coolwarm', aspect='auto')
#                     ax.set_title(f"Explanation Difference (Round {round2} - Round {round1})")
#                     plt.colorbar(im, ax=ax)
#                     st.pyplot(fig)
                    
#                     st.markdown("""
#                     **Difference Interpretation**:
#                     - Red regions: Features that gained importance in the later round
#                     - Blue regions: Features that lost importance in the later round
#                     - White regions: Features with similar importance in both rounds
#                     """)
        
#         except Exception as e:
#             st.error(f"Error comparing rounds: {e}")

# Dataset Viewer Tab
with tab3:
    st.markdown("<h2 class='sub-header'>Dataset Viewer</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Browse through the dataset samples, visualize them, and manually verify model predictions.
    </div>
    """, unsafe_allow_html=True)
    
    # Client and data split selection
    col1, col2 = st.columns(2)
    with col1:
        viewer_client_id = st.selectbox("Select client", list(range(20)), key="viewer_client")
    with col2:
        viewer_data_split = st.radio("Data split", ["test", "train"], key="viewer_split")
    
    # Try different possible paths for client data
    possible_paths = [
        f"../clients_data/client_{viewer_client_id}_{viewer_data_split}.npz",  # Up one level
        f"clients_data/client_{viewer_client_id}_{viewer_data_split}.npz",      # Same level
        f"../../clients_data/client_{viewer_client_id}_{viewer_data_split}.npz"  # Up two levels
    ]
    
    sample_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sample_path = path
            break
    
    if sample_path:
        # Load data
        try:
            data = np.load(sample_path)
            X_all_viewer = data['X']
            y_all_viewer = data['y']
            num_samples = len(y_all_viewer)
            
            st.success(f"Loaded {num_samples} samples from {sample_path}")
            
            # Navigation controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous"):
                    if "viewer_sample_idx" in st.session_state and st.session_state.viewer_sample_idx > 0:
                        st.session_state.viewer_sample_idx -= 1
            
            with col2:
                # Initialize session state if not exists
                if "viewer_sample_idx" not in st.session_state:
                    st.session_state.viewer_sample_idx = 0
                
                viewer_sample_idx = st.slider("Sample index", 0, num_samples-1, 
                                            st.session_state.viewer_sample_idx, key="viewer_sample_idx")
            
            with col3:
                if st.button("Next ➡️"):
                    if st.session_state.viewer_sample_idx < num_samples - 1:
                        st.session_state.viewer_sample_idx += 1
            
            # Get current sample
            X_sample_viewer = X_all_viewer[viewer_sample_idx]
            y_true_viewer = int(y_all_viewer[viewer_sample_idx])
            activity_id = IDX_TO_ACTIVITY.get(y_true_viewer, y_true_viewer)
            true_activity = ACTIVITY_LABELS.get(activity_id, 'Unknown')
            
            # Display ground truth
            st.info(f"Ground truth label: {y_true_viewer} ({true_activity})")
            
            # Make prediction
            X_input_viewer = X_sample_viewer[np.newaxis, ...]
            prediction_viewer = model.predict(X_input_viewer, verbose=0)
            pred_class_viewer = np.argmax(prediction_viewer, axis=1)[0]
            pred_prob_viewer = prediction_viewer[0, pred_class_viewer]
            pred_activity_id = IDX_TO_ACTIVITY.get(pred_class_viewer, pred_class_viewer)
            pred_activity = ACTIVITY_LABELS.get(pred_activity_id, 'Unknown')
            
            # Display prediction
            if pred_class_viewer == y_true_viewer:
                st.success(f"✅ Correct prediction: {pred_class_viewer} ({pred_activity}) with {pred_prob_viewer:.2f} confidence")
            else:
                st.error(f"❌ Wrong prediction: {pred_class_viewer} ({pred_activity}) with {pred_prob_viewer:.2f} confidence")
            
            # # Manual verification
            # st.markdown("### Manual Verification")
            # manual_verification = st.radio(
            #     "Is the prediction correct?", 
            #     ["Yes", "No", "Unsure"],
            #     key=f"verification_{viewer_client_id}_{viewer_data_split}_{viewer_sample_idx}"
            # )
            
            # # Visualization tabs
            # viz_tab1, viz_tab2 = st.tabs(["Time Series Plot", "Heatmap"])
            
            # with viz_tab1:
            #     # Create a DataFrame for better plotting - dynamically adapt to the data shape
            #     num_features = X_sample_viewer.shape[1]
                
            #     if len(SENSOR_COLUMNS) != num_features:
            #         # If SENSOR_COLUMNS doesn't match, create generic column names
            #         st.warning(f"SENSOR_COLUMNS length ({len(SENSOR_COLUMNS)}) doesn't match data features ({num_features}). Using generic column names.")
            #         column_names = [f"feature_{i}" for i in range(num_features)]
            #     else:
            #         column_names = SENSOR_COLUMNS
                
            #     df = pd.DataFrame(X_sample_viewer, columns=column_names)
            #     df['timestep'] = range(len(df))
                
            #     # Reshape for plotting
            #     df_melted = pd.melt(df, id_vars=['timestep'], value_vars=column_names,
            #                        var_name='sensor', value_name='value')
                
            #     # Group sensors based on name patterns or just plot everything together
            #     if len(SENSOR_COLUMNS) == num_features:
            #         # We can use the original location/sensor grouping
            #         df_melted[['location', 'sensor_type']] = df_melted['sensor'].str.split('_', n=1, expand=True)
                    
            #         # Plot by location
            #         locations = df_melted['location'].unique()
            #         for loc in locations:
            #             st.subheader(f"{loc.capitalize()} Sensors")
            #             loc_data = df_melted[df_melted['location'] == loc]
                        
            #             fig = px.line(loc_data, x='timestep', y='value', color='sensor_type',
            #                        title=f"{loc.capitalize()} Sensor Data",
            #                        labels={'timestep': 'Time', 'value': 'Sensor Value', 'sensor_type': 'Sensor Type'})
            #             st.plotly_chart(fig, use_container_width=True)
            #     else:
            #         # Plot all sensors together in groups of 6 to avoid overcrowding
            #         num_groups = (num_features + 5) // 6  # Ceiling division
            #         for group_idx in range(num_groups):
            #             start_idx = group_idx * 6
            #             end_idx = min(start_idx + 6, num_features)
            #             group_cols = column_names[start_idx:end_idx]
                        
            #             st.subheader(f"Feature Group {group_idx+1}")
            #             group_data = df_melted[df_melted['sensor'].isin(group_cols)]
                        
            #             fig = px.line(group_data, x='timestep', y='value', color='sensor',
            #                        title=f"Sensor Data - Group {group_idx+1}",
            #                        labels={'timestep': 'Time', 'value': 'Sensor Value', 'sensor': 'Feature'})
            #             st.plotly_chart(fig, use_container_width=True)
            
            # with viz_tab2:
            #     # Create a heatmap of the sensor data with dynamic y-axis labels
            #     if len(SENSOR_COLUMNS) == X_sample_viewer.shape[1]:
            #         y_labels = SENSOR_COLUMNS
            #     else:
            #         y_labels = [f"feature_{i}" for i in range(X_sample_viewer.shape[1])]
                
            #     fig = px.imshow(X_sample_viewer.T,
            #                    labels=dict(x="Time Step", y="Sensor", color="Value"),
            #                    x=list(range(X_sample_viewer.shape[0])),
            #                    y=y_labels,
            #                    aspect="auto",
            #                    title="Sensor Data Heatmap")
            #     st.plotly_chart(fig, use_container_width=True)
            
            # # Generate explanation if requested
            # if st.button("Generate Explanation", key=f"gen_exp_btn_{viewer_client_id}_{viewer_data_split}_{viewer_sample_idx}"):
            #     xai_method = st.radio("Explanation Method", ["GradCAM", "Integrated Gradients"], key=f"viewer_xai_method_{viewer_client_id}_{viewer_data_split}_{viewer_sample_idx}")
            #     st.info(f"Generating {xai_method} explanation...")
                
            #     try:
            #         # Force model initialization with this exact sample shape
            #         dummy_input = np.zeros((1,) + X_sample_viewer.shape)
            #         _ = model.predict(dummy_input, verbose=0)
                    
            #         if xai_method == "GradCAM":
            #             explanation, _ = get_gradcam_explanation(model, X_input_viewer, pred_class_viewer)
            #         else:  # Integrated Gradients
            #             explanation, _ = get_integrated_gradients_explanation(model, X_input_viewer, pred_class_viewer)
                    
            #         # Normalize explanation
            #         explanation = explanation / np.max(np.abs(explanation)) if np.max(np.abs(explanation)) > 0 else explanation
                    
            #         # Plot explanation with dynamic y-axis labels
            #         if len(SENSOR_COLUMNS) == explanation.shape[1]:
            #             y_labels = SENSOR_COLUMNS
            #         else:
            #             y_labels = [f"feature_{i}" for i in range(explanation.shape[1])]
                        
            #         fig = px.imshow(explanation.T,
            #                        labels=dict(x="Time Step", y="Sensor", color="Attribution"),
            #                        x=list(range(explanation.shape[0])),
            #                        y=y_labels,
            #                        aspect="auto",
            #                        color_continuous_scale="RdBu_r",
            #                        title=f"{xai_method} Explanation for Prediction")
            #         st.plotly_chart(fig, use_container_width=True)
                    
            #         # Generate plotly time-series attribution plot
            #         st.subheader("Time Series with Attribution Overlay")
                    
            #         # Use dynamic column names based on data shape
            #         if len(SENSOR_COLUMNS) == X_sample_viewer.shape[1]:
            #             feature_names = SENSOR_COLUMNS
            #         else:
            #             feature_names = [f"feature_{i}" for i in range(X_sample_viewer.shape[1])]
                        
            #         ts_fig = plot_time_series_attribution(X_sample_viewer, explanation, feature_names)
            #         st.plotly_chart(ts_fig, use_container_width=True)
                    
            #     except Exception as e:
            #         st.error(f"Error generating explanation: {e}")
            #         st.write("Error details:", str(e))
                
        except Exception as e:
            st.error(f"Error loading or processing data: {e}")
    else:
        st.error(f"Could not find client data file for client {viewer_client_id}. Please check the data paths.")

# Display information about the dataset and model in an expandable section
with st.expander("About the Dataset and Model"):
    st.markdown("""
    ### PAMAP2 Dataset
    
    The PAMAP2 dataset contains data from 9 subjects performing 18 different physical activities,
    with data collected from IMU sensors on the hand, chest, and ankle.
    
    **Features:**
    - Accelerometer (3 axes)
    - Gyroscope (3 axes)
    - Magnetometer (3 axes, not used in this model)
    
    **Activities in this model:**
    - Lying (1)
    - Sitting (2)
    - Standing (3)
    - Walking (4)
    - Running (5)
    - Cycling (6)
    - Nordic walking (7)
    - Ascending stairs (12)
    - Descending stairs (13)
    - Vacuum cleaning (16)
    
    ### FedBN Model Architecture
    
    This model uses a Convolutional Neural Network with BatchNormalization for activity recognition:
    
    - Input: 200 timesteps with sensor features
    - Conv1D layer 1: 64 filters, kernel=3, ReLU, MaxPool(2)
    - Conv1D layer 2: 128 filters, kernel=3, ReLU, MaxPool(2)
    - Dense layers: 128 units → 10 classes
    
    The model is trained using Federated Learning with BatchNormalization (FedBN),
    where clients share model parameters except for BatchNormalization layers.
    """)

# Footer
st.markdown("---")
st.markdown("FedBN Activity Recognition - Explainable AI Dashboard")
