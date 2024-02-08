import gradio as gr
from gradio.components import Textbox
import requests

def predict(item_id):
    # Call FastAPI endpoint to get item details from MongoDB
    response = requests.get(f"http://127.0.0.1:8000/students/{item_id}")
    if response.status_code == 200:
        item_data = response.json()
        return f"Item Name: {item_data['name']}\nDescription: {item_data['course']}"
    return "Item not found"

if __name__ == "__main__":
    iface = gr.Interface(
        fn=predict,
        inputs=Textbox(label="Item ID"),
        outputs=Textbox(label="Item Details"),
    )
    iface.launch()