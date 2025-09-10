import gradio as gr
import joblib
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("best_model_pipeline.joblib")

# -----------------------------
# Prediction function
# -----------------------------
def predict_review(delivery_delay_days, shipping_time_days, purchase_dow, purchase_hour):
    # Put inputs into a DataFrame
    input_data = pd.DataFrame({
        "delivery_delay_days": [delivery_delay_days],
        "shipping_time_days": [shipping_time_days],
        "purchase_dow": [purchase_dow],
        "purchase_hour": [purchase_hour]
    })
    
    # Get probability of negative review
    probability = model.predict_proba(input_data)[:, 1][0]
    
    # Format result
    result = f"The model predicts a {probability*100:.2f}% probability that this client will leave a NEGATIVE review (≤2★)."
    return result

# -----------------------------
# Build Gradio interface
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Olist Negative Review Predictor (Prototype)")
    gr.Markdown("Enter order details below to predict the probability of receiving a **negative review (≤2★)**.")
    
    with gr.Row():
        purchase_dow = gr.Dropdown(choices=list(range(7)), label="Purchase Day of Week (0=Mon … 6=Sun)", value=0)
        purchase_hour = gr.Slider(0, 23, value=12, step=1, label="Purchase Hour (0–23)")
    
    with gr.Row():
        delivery_delay_days = gr.Number(label="Delivery Delay (days)", value=0)
        shipping_time_days = gr.Number(label="Shipping Time (days)", value=5)
    
    predict_btn = gr.Button("🔮 Predict")
    output_text = gr.Textbox(label="Prediction Result", lines=2)
    
    predict_btn.click(
        fn=predict_review,
        inputs=[purchase_dow, purchase_hour, delivery_delay_days, shipping_time_days],
        outputs=output_text
    )

# -----------------------------
# Launch app
# -----------------------------
if __name__ == "__main__":
    demo.launch()  # Set share=True to get a public link