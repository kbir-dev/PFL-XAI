import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import json
import os

st.set_page_config(page_title="PFL-XAI: Feature Importance Dashboard", layout="wide")

# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    # Load model without compiling (avoids optimizer issues)
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y']

def get_pamap2_feature_names():
    """Return the 42 PAMAP2 feature names after preprocessing (removed timestamp, heart_rate, and magnetometer columns)"""
    feature_names = [
        # Hand IMU (11 features)
        'hand_temp', 'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
        'hand_acc6_x', 'hand_acc6_y', 'hand_acc6_z',
        'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
        'hand_orientation_1', 'hand_orientation_2', 'hand_orientation_3', 'hand_orientation_4',
        
        # Chest IMU (11 features)
        'chest_temp', 'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
        'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
        'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
        'chest_orientation_1', 'chest_orientation_2', 'chest_orientation_3', 'chest_orientation_4',
        
        # Ankle IMU (11 features)
        'ankle_temp', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
        'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
        'ankle_orientation_1', 'ankle_orientation_2', 'ankle_orientation_3', 'ankle_orientation_4'
    ]
    return feature_names

def get_activity_class_names():
    """Return the PAMAP2 activity class names (mapped from keep_classes [1,2,3,4,5,6,7,12,13,16] to indices 0-9)"""
    activity_names = {
        0: "Lying",
        1: "Sitting", 
        2: "Standing",
        3: "Walking",
        4: "Running",
        5: "Cycling",
        6: "Nordic Walking",
        7: "Ascending Stairs",
        8: "Descending Stairs",
        9: "Vacuum Cleaning"
    }
    return activity_names

def compute_gradient_attribution(model, sample):
    """Compute gradient-based feature importance (fast method)"""
    sample_tensor = tf.constant(sample, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(sample_tensor)
        predictions = model(sample_tensor)
        target_score = tf.reduce_max(predictions)
    
    gradients = tape.gradient(target_score, sample_tensor)
    return gradients

def plot_feature_importance(feature_importance, pred_class):
    """Plot feature importance bar chart"""
    # Get feature names (custom or PAMAP2 default)
    num_features = len(feature_importance)
    if 'feature_names' in st.session_state and st.session_state['feature_names'] is not None:
        feature_names = st.session_state['feature_names']
    else:
        feature_names = get_pamap2_feature_names()
    
    # Get activity name for title
    activity_names = get_activity_class_names()
    activity_name = activity_names.get(pred_class, f"Unknown ({pred_class})")
    
    # Plot feature importance bar chart
    st.subheader("📊 Feature Importance Analysis")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create colormap for different colors
    colors = cm.rainbow(np.linspace(0, 1, num_features))
    
    bars = ax.bar(range(num_features), feature_importance, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set feature names on x-axis
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
    ax.set_xlabel('Feature Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score (Avg. Absolute Gradient)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance for {activity_name} (Class {pred_class})', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Use scientific notation for y-axis if values are very small
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Highlight top 10 most important features with labels
    top_10_indices = np.argsort(feature_importance)[-10:]
    for idx in top_10_indices:
        height = feature_importance[idx]
        # Use scientific notation for small values
        if height < 0.001:
            label = f'{height:.2e}'
        else:
            label = f'{height:.4f}'
        ax.text(idx, height, label, ha='center', va='bottom', fontsize=7, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show top features summary with better formatting
    st.markdown("### 🏆 Top 10 Most Important Features")
    
    # Create a more detailed dataframe
    top_10_sorted = np.argsort(feature_importance)[-10:][::-1]
    top_features_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Feature Name": [feature_names[i] for i in top_10_sorted],
        "Feature Index": top_10_sorted,
        "Importance Score": feature_importance[top_10_sorted],
        "Percentage of Max": (feature_importance[top_10_sorted] / feature_importance.max() * 100)
    })
    
    # Format the dataframe
    st.dataframe(
        top_features_df.style.format({
            "Importance Score": "{:.6f}",
            "Percentage of Max": "{:.2f}%"
        }).background_gradient(subset=['Importance Score'], cmap='RdYlGn'),
        use_container_width=True
    )

def load_json_metrics(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# -------------------------------------------------------------
# Get script directory for relative paths
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------
# Streamlit Layout
# -------------------------------------------------------------

st.title("🧠 PFL-XAI Dashboard (Feature Importance Analysis)")
tabs = st.tabs(["Single Sample Analysis", "Client Data Analysis", "Round Comparison"])

# -------------------------------------------------------------
# 1️⃣ Single Sample Analysis
# -------------------------------------------------------------
with tabs[0]:
    st.header("🔹 Single Sample Feature Importance Analysis")

    # Model Selection
    st.subheader("📁 Select Model")
    model_source = st.radio("Choose model source:", ["Select from available models", "Upload custom model"], horizontal=True)
    
    model_path = None
    if model_source == "Select from available models":
        models_dir = os.path.join(SCRIPT_DIR, "models")
        if os.path.exists(models_dir):
            model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".h5")], 
                                key=lambda x: int(x.split("_")[-1].replace(".h5", "")) if "round" in x else 0)
            if model_files:
                selected_model = st.selectbox("Select a model:", model_files, index=len(model_files)-1)
                model_path = os.path.join(models_dir, selected_model)
                st.success(f"✅ Selected: {selected_model}")
            else:
                st.warning("No models found in the models directory.")
        else:
            st.warning("Models directory not found.")
    else:
        uploaded_model = st.file_uploader("Upload trained model (.h5)", type=["h5"])
        if uploaded_model:
            model_path = uploaded_model
    
    # Sample Selection
    st.subheader("📊 Select Test Sample")
    sample_source = st.radio("Choose sample source:", ["Select from client data", "Upload custom sample"], horizontal=True, key="sample_radio")
    
    sample_file = None
    if sample_source == "Select from client data":
        clients_dir = os.path.join(SCRIPT_DIR, "clients_data")
        if os.path.exists(clients_dir):
            sample_files = sorted([f for f in os.listdir(clients_dir) if f.endswith("_test.npz")])
            if sample_files:
                selected_sample = st.selectbox("Select a test sample:", sample_files)
                sample_file = os.path.join(clients_dir, selected_sample)
                st.success(f"✅ Selected: {selected_sample}")
            else:
                st.warning("No test samples found in clients_data directory.")
        else:
            st.warning("Clients data directory not found.")
    else:
        uploaded_sample = st.file_uploader("Upload test sample (.npz)", type=["npz"], key="sample_upload")
        if uploaded_sample:
            sample_file = uploaded_sample

    # Feature Importance Analysis
    if model_path and sample_file:
        st.markdown("---")
        st.subheader("🔍 Feature Importance Results")
        
        model = load_model(model_path)
        X, y = load_npz(sample_file)

        st.write(f"**Dataset shape:** {X.shape}, **Labels:** {len(np.unique(y))} classes")
        idx = st.slider("Select sample index", 0, len(X) - 1, 0)
        sample = np.expand_dims(X[idx], axis=0)

        # Get activity class names
        activity_names = get_activity_class_names()
        true_label = int(y[idx])
        true_activity = activity_names.get(true_label, f"Unknown ({true_label})")
        
        # Prediction
        pred_probs = model.predict(sample, verbose=0)
        pred = np.argmax(pred_probs)
        pred_activity = activity_names.get(pred, f"Unknown ({pred})")
        pred_confidence = pred_probs[0][pred] * 100
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True Label", f"{true_label}: {true_activity}")
        with col2:
            st.metric(
                "Predicted Class", 
                f"{pred}: {pred_activity}",
                delta="✓ Correct" if pred == true_label else "✗ Wrong"
            )
        with col3:
            st.metric("Confidence", f"{pred_confidence:.2f}%")
        
        # Show all activity classes for reference
        with st.expander("📋 View All Activity Classes"):
            activity_df = {
                "Class ID": list(activity_names.keys()),
                "Activity Name": list(activity_names.values())
            }
            st.dataframe(activity_df, use_container_width=True, hide_index=True)

        # Feature Names Configuration
        st.markdown("---")
        
        # Option to provide custom feature names
        with st.expander("📝 Feature Information (Optional)"):
            st.info("**Default PAMAP2 Features (42):**\n"
                   "- Hand IMU: temp, acc16_x/y/z, acc6_x/y/z, gyro_x/y/z, orientation_1/2/3/4\n"
                   "- Chest IMU: temp, acc16_x/y/z, acc6_x/y/z, gyro_x/y/z, orientation_1/2/3/4\n"
                   "- Ankle IMU: temp, acc16_x/y/z, acc6_x/y/z, gyro_x/y/z, orientation_1/2/3/4")
            
            st.write("**Removed features:** timestamp, heart_rate, magnetometer readings (hand/chest/ankle)")
            
            st.markdown("---")
            st.write("**Override with custom names (optional):**")
            custom_names_input = st.text_area(
                "Enter 42 custom feature names (comma-separated):",
                placeholder="e.g., Feature_A, Feature_B, Feature_C, ..."
            )
            
            if custom_names_input:
                custom_names = [name.strip() for name in custom_names_input.split(',')]
                if len(custom_names) == X.shape[2]:
                    st.success(f"✅ Using {len(custom_names)} custom feature names")
                    st.session_state['feature_names'] = custom_names
                else:
                    st.error(f"❌ Expected {X.shape[2]} feature names but got {len(custom_names)}")
                    st.session_state['feature_names'] = None
            else:
                st.session_state['feature_names'] = None
        
        # Feature Importance Analysis (automatic)
        st.markdown("---")
        st.subheader("🔍 Feature Importance Analysis")
        st.info("⚡ **Using Fast Gradient Attribution** - Instant results based on model gradients!")
        
        with st.spinner("⚡ Computing feature importance..."):
            gradients = compute_gradient_attribution(model, sample)
            
            if gradients is not None:
                st.success("✅ Analysis complete!")
                
                # Calculate feature importance
                grad_array = gradients.numpy()[0]
                feature_importance = np.mean(np.abs(grad_array), axis=0)
                
                # Plot feature importance
                plot_feature_importance(feature_importance, pred)
            else:
                st.error("Could not compute gradients. Please try with a different model or sample.")

# -------------------------------------------------------------
# 2️⃣ Client Data Analysis
# -------------------------------------------------------------
with tabs[1]:
    st.header("🏢 Client-wise Metrics Viewer")

    client_dir = st.text_input("Enter client data directory path", os.path.join(SCRIPT_DIR, "fedbn", "metrics"))
    if os.path.exists(client_dir):
        # Filter out federated metrics, only show client metrics
        metrics_files = sorted([f for f in os.listdir(client_dir) if f.endswith(".json") and "client" in f])
        if metrics_files:
            selected = st.selectbox("Select a client metrics file", metrics_files)
            data = load_json_metrics(os.path.join(client_dir, selected))
            if data:
                st.subheader(f"📊 Metrics for {selected}")
                
                # Display metrics in a nice format
                if "client_id" in data:
                    st.write(f"**Client ID:** {data['client_id']}")
                
                # Create columns for train and test metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Training Metrics")
                    if "train_accuracy" in data:
                        st.metric("Train Accuracy", f"{data['train_accuracy']:.4f}")
                    if "train_loss" in data:
                        st.metric("Train Loss", f"{data['train_loss']:.4f}")
                
                with col2:
                    st.markdown("### ✅ Test Metrics")
                    if "test_accuracy" in data:
                        st.metric("Test Accuracy", f"{data['test_accuracy']:.4f}")
                    if "test_loss" in data:
                        st.metric("Test Loss", f"{data['test_loss']:.4f}")
                
                # Show raw JSON
                with st.expander("View Raw JSON"):
                    st.json(data)
            else:
                st.warning("No metrics found in this file.")
        else:
            st.info("No client metrics files found in the directory.")
    else:
        st.warning("Directory not found.")

# -------------------------------------------------------------
# 3️⃣ Round Comparison
# -------------------------------------------------------------
with tabs[2]:
    st.header("🔁 Federated Round Comparison")

    # Default path to federated metrics file
    default_metrics_path = os.path.join(SCRIPT_DIR, "fedbn", "metrics", "fedbn_metrics.json")
    metrics_file_path = st.text_input("Enter federated metrics file path", default_metrics_path)
    
    if os.path.exists(metrics_file_path):
        data = load_json_metrics(metrics_file_path)
        if data and "accuracy" in data and "loss" in data:
            acc = data["accuracy"]
            loss = data["loss"]
            
            if len(acc) > 0:
                st.subheader("📈 Round-wise Accuracy")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(acc, marker="o", color="green", linewidth=2)
                ax1.set_xlabel("Round")
                ax1.set_ylabel("Accuracy")
                ax1.set_title("Federated Learning - Accuracy per Round")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close()
                
                st.subheader("📉 Round-wise Loss")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(loss, marker="o", color="red", linewidth=2)
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Loss")
                ax2.set_title("Federated Learning - Loss per Round")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                plt.close()
                
                # Show summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Accuracy", f"{acc[-1]:.4f}")
                    st.metric("Best Accuracy", f"{max(acc):.4f}")
                    st.metric("Average Accuracy", f"{np.mean(acc):.4f}")
                with col2:
                    st.metric("Final Loss", f"{loss[-1]:.4f}")
                    st.metric("Best Loss", f"{min(loss):.4f}")
                    st.metric("Average Loss", f"{np.mean(loss):.4f}")
            else:
                st.warning("No round data found in metrics file.")
        else:
            st.warning("Metrics file doesn't contain 'accuracy' and 'loss' fields.")
    else:
        st.warning("Metrics file not found. Please check the path.")
