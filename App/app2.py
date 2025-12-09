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
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_npz(path):
    data = np.load(path)
    return data['X'], data['y']

def get_pamap2_feature_names():
    """Return the 42 PAMAP2 feature names after preprocessing"""
    feature_names = [
        'hand_temp', 'hand_acc16_x', 'hand_acc16_y', 'hand_acc16_z',
        'hand_acc6_x', 'hand_acc6_y', 'hand_acc6_z',
        'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
        'hand_orientation_1', 'hand_orientation_2', 'hand_orientation_3', 'hand_orientation_4',

        'chest_temp', 'chest_acc16_x', 'chest_acc16_y', 'chest_acc16_z',
        'chest_acc6_x', 'chest_acc6_y', 'chest_acc6_z',
        'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
        'chest_orientation_1', 'chest_orientation_2', 'chest_orientation_3', 'chest_orientation_4',

        'ankle_temp', 'ankle_acc16_x', 'ankle_acc16_y', 'ankle_acc16_z',
        'ankle_acc6_x', 'ankle_acc6_y', 'ankle_acc6_z',
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
        'ankle_orientation_1', 'ankle_orientation_2', 'ankle_orientation_3', 'ankle_orientation_4'
    ]
    return feature_names

def get_activity_class_names():
    return {
        0: "Lying", 1: "Sitting", 2: "Standing", 3: "Walking",
        4: "Running", 5: "Cycling", 6: "Nordic Walking",
        7: "Ascending Stairs", 8: "Descending Stairs", 9: "Vacuum Cleaning"
    }

# -------------------------------------------------------------
# 🌈 Improved XAI Method: Integrated Gradients
# -------------------------------------------------------------

def compute_integrated_gradients(model, sample, baseline=None, steps=50):
    """Compute Integrated Gradients for a given input sample."""
    sample = tf.convert_to_tensor(sample, dtype=tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(sample)

    interpolated_inputs = [
        baseline + (float(i) / steps) * (sample - baseline)
        for i in range(steps + 1)
    ]
    interpolated_inputs = tf.concat(interpolated_inputs, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        preds = model(interpolated_inputs)
        target_class = tf.argmax(preds[-1])
        target_output = preds[:, target_class]

    grads = tape.gradient(target_output, interpolated_inputs)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (sample - baseline)[0] * avg_grads

    return integrated_grads.numpy()

# -------------------------------------------------------------
# Visualization
# -------------------------------------------------------------

def plot_feature_importance(feature_importance, pred_class):
    feature_names = st.session_state.get('feature_names', get_pamap2_feature_names())
    activity_names = get_activity_class_names()
    activity_name = activity_names.get(pred_class, f"Unknown ({pred_class})")

    # Remove redundant features
    ignore_features = {'hand_temp', 'chest_temp', 'ankle_temp'}
    mask = [f not in ignore_features for f in feature_names]
    feature_importance = feature_importance[mask]
    feature_names = [f for f in feature_names if f not in ignore_features]

    st.subheader("📊 Feature Importance Analysis (Integrated Gradients)")
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = cm.rainbow(np.linspace(0, 1, len(feature_importance)))
    ax.bar(range(len(feature_importance)), feature_importance, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(feature_importance)))
    ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
    ax.set_xlabel('Feature Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance for {activity_name} (Class {pred_class})', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    top_10_indices = np.argsort(feature_importance)[-10:]
    for idx in top_10_indices:
        height = feature_importance[idx]
        label = f'{height:.4f}' if height >= 0.001 else f'{height:.2e}'
        ax.text(idx, height, label, ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### 🏆 Top 10 Most Important Features")
    top_10_sorted = np.argsort(feature_importance)[-10:][::-1]
    top_features_df = pd.DataFrame({
        "Rank": range(1, 11),
        "Feature Name": [feature_names[i] for i in top_10_sorted],
        "Feature Index": top_10_sorted,
        "Importance Score": feature_importance[top_10_sorted],
        "Percentage of Max": (feature_importance[top_10_sorted] / feature_importance.max() * 100)
    })
    st.dataframe(
        top_features_df.style.format({
            "Importance Score": "{:.6f}",
            "Percentage of Max": "{:.2f}%"
        }).background_gradient(subset=['Importance Score'], cmap='RdYlGn'),
        use_container_width=True
    )

# -------------------------------------------------------------
# JSON Metrics Loader
# -------------------------------------------------------------

def load_json_metrics(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# -------------------------------------------------------------
# Streamlit Layout
# -------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
st.title("🧠 PFL-XAI Dashboard (Integrated Gradients Edition)")
tabs = st.tabs(["Single Sample Analysis", "Client Data Analysis", "Round Comparison"])

# -------------------------------------------------------------
# 1️⃣ Single Sample Analysis
# -------------------------------------------------------------
with tabs[0]:
    st.header("🔹 Single Sample Feature Importance Analysis")

    # Model
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
                st.warning("No models found.")
        else:
            st.warning("Models directory not found.")
    else:
        uploaded_model = st.file_uploader("Upload trained model (.h5)", type=["h5"])
        if uploaded_model:
            model_path = uploaded_model

    # Sample
    st.subheader("📊 Select Test Sample")
    sample_source = st.radio("Choose sample source:", ["Select from client data", "Upload custom sample"], horizontal=True)

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
                st.warning("No test samples found.")
        else:
            st.warning("Clients data directory not found.")
    else:
        uploaded_sample = st.file_uploader("Upload test sample (.npz)", type=["npz"])
        if uploaded_sample:
            sample_file = uploaded_sample

    # Run XAI
    if model_path and sample_file:
        st.markdown("---")
        st.subheader("🔍 Feature Importance Results")

        model = load_model(model_path)
        X, y = load_npz(sample_file)

        st.write(f"**Dataset shape:** {X.shape}, **Labels:** {len(np.unique(y))} classes")
        idx = st.slider("Select sample index", 0, len(X) - 1, 0)
        sample = np.expand_dims(X[idx], axis=0)

        activity_names = get_activity_class_names()
        true_label = int(y[idx])
        true_activity = activity_names.get(true_label, f"Unknown ({true_label})")

        pred_probs = model.predict(sample, verbose=0)
        pred = np.argmax(pred_probs)
        pred_activity = activity_names.get(pred, f"Unknown ({pred})")
        pred_confidence = pred_probs[0][pred] * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True Label", f"{true_label}: {true_activity}")
        with col2:
            st.metric("Predicted Class", f"{pred}: {pred_activity}", delta="✓ Correct" if pred == true_label else "✗ Wrong")
        with col3:
            st.metric("Confidence", f"{pred_confidence:.2f}%")

        st.markdown("---")
        st.subheader("⚙️ Explainability Settings")
        steps = st.slider("Interpolation Steps (Higher = Smoother)", 10, 200, 50, step=10)

        with st.spinner("⚡ Computing Integrated Gradients..."):
            ig = compute_integrated_gradients(model, sample, steps=steps)
            feature_importance = np.mean(np.abs(ig), axis=0)
            feature_importance = feature_importance / np.sum(feature_importance)

            st.success("✅ Explanation Ready!")
            plot_feature_importance(feature_importance, pred)

# -------------------------------------------------------------
# 2️⃣ Client Data Analysis
# -------------------------------------------------------------
with tabs[1]:
    st.header("🏢 Client-wise Metrics Viewer")

    client_dir = st.text_input("Enter client data directory path", os.path.join(SCRIPT_DIR, "fedbn", "metrics"))
    if os.path.exists(client_dir):
        metrics_files = sorted([f for f in os.listdir(client_dir) if f.endswith(".json") and "client" in f])
        if metrics_files:
            selected = st.selectbox("Select a client metrics file", metrics_files)
            data = load_json_metrics(os.path.join(client_dir, selected))
            if data:
                st.subheader(f"📊 Metrics for {selected}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Accuracy", f"{data.get('train_accuracy', 0):.4f}")
                    st.metric("Train Loss", f"{data.get('train_loss', 0):.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{data.get('test_accuracy', 0):.4f}")
                    st.metric("Test Loss", f"{data.get('test_loss', 0):.4f}")

                with st.expander("View Raw JSON"):
                    st.json(data)
            else:
                st.warning("No metrics found.")
        else:
            st.info("No client metrics files found.")
    else:
        st.warning("Directory not found.")

# -------------------------------------------------------------
# 3️⃣ Round Comparison
# -------------------------------------------------------------
with tabs[2]:
    st.header("🔁 Federated Round Comparison")

    default_metrics_path = os.path.join(SCRIPT_DIR, "fedbn", "metrics", "fedbn_metrics.json")
    metrics_file_path = st.text_input("Enter federated metrics file path", default_metrics_path)

    if os.path.exists(metrics_file_path):
        data = load_json_metrics(metrics_file_path)
        if data and "accuracy" in data and "loss" in data:
            acc, loss = data["accuracy"], data["loss"]
            if len(acc) > 0:
                st.subheader("📈 Round-wise Accuracy")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(acc, marker="o", color="green", linewidth=2)
                ax1.set_xlabel("Round"); ax1.set_ylabel("Accuracy")
                ax1.set_title("Federated Learning - Accuracy per Round")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)

                st.subheader("📉 Round-wise Loss")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(loss, marker="o", color="red", linewidth=2)
                ax2.set_xlabel("Round"); ax2.set_ylabel("Loss")
                ax2.set_title("Federated Learning - Loss per Round")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

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
            st.warning("Metrics file missing accuracy/loss fields.")
    else:
        st.warning("Metrics file not found.")
