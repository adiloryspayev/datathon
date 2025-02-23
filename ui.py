import tkinter as tk
import random
import time
import pandas as pd

# Function to simulate address to latitude and longitude conversion
def address_to_lat_lng(address):
    lat = random.uniform(-90, 90)
    lng = random.uniform(-180, 180)
    return lat, lng

# Load the dataset and extract unique service types from 'Summary' column
def get_service_types():
    df = pd.read_csv("cityline.csv")  
    unique_service_types = df['Summary'].unique()
    return unique_service_types

# Function to simulate prediction
def make_prediction():
    address = address_entry.get().strip()
    
    if not address:
        result_service_label.config(text="⚠️ Please enter an address.", fg="red")
        result_time_label.config(text="")
        result_frame.pack(pady=20)
        return
    
    # Show loading screen
    loading_screen.pack(pady=10)
    root.update_idletasks()  # Forces UI to update

    time.sleep(2)  # Simulate processing time
    
    # Generate random predictions
    service_type = random.choice(service_types)
    time_attention = random.choice(time_to_attention)

    # Display results
    loading_screen.pack_forget()
    result_service_label.config(text=f"✅ Predicted Service: {service_type}", fg="#2E86C1")
    result_time_label.config(text=f"⏳ Time to Attention: {time_attention}", fg="#117A65")
    result_frame.pack(pady=15)

# Define time attention labels
time_to_attention = ['Soon', 'Medium', 'Later']
service_types = get_service_types()

# Setup UI window
root = tk.Tk()
root.title("ML Service Prediction")
root.geometry("500x350")
root.configure(bg="#F9F9F9")

# Style Config
FONT_HEADER = ("Helvetica", 16, "bold")
FONT_TEXT = ("Helvetica", 12)
FONT_BUTTON = ("Helvetica", 12, "bold")

# Header
heading_label = tk.Label(root, text="📍 Enter Address to Predict Service", font=FONT_HEADER, bg="#F9F9F9", fg="#333")
heading_label.pack(pady=15)

# Input Field
address_entry = tk.Entry(root, width=40, font=FONT_TEXT, relief="solid", bd=1)
address_entry.pack(pady=10, ipady=5)

# Predict Button (Styled)
def on_hover(event):
    predict_button.config(bg="#1E8449", cursor="hand2")

def on_leave(event):
    predict_button.config(bg="#229954")

predict_button = tk.Button(root, text="🔍 Predict Service", font=FONT_BUTTON, width=20, height=2, 
                           bg="#229954", fg="white", relief="flat", command=make_prediction)
predict_button.pack(pady=10)
predict_button.bind("<Enter>", on_hover)
predict_button.bind("<Leave>", on_leave)

# Loading Screen (Hidden Initially)
loading_screen = tk.Label(root, text="⏳ Processing... Please wait.", font=FONT_TEXT, fg="#E67E22", bg="#F9F9F9")
loading_screen.pack_forget()

# Result Frame (Hidden Initially)
result_frame = tk.Frame(root, bg="#F9F9F9")

result_service_label = tk.Label(result_frame, text="Predicted Service: ", font=FONT_TEXT, bg="#F9F9F9", fg="#333")
result_service_label.pack()

result_time_label = tk.Label(result_frame, text="Time to Attention: ", font=FONT_TEXT, bg="#F9F9F9", fg="#333")
result_time_label.pack()

# Footer
footer_label = tk.Label(root, text="🔹 Powered by PyTorch 🔹", font=("Helvetica", 10, "italic"), bg="#F9F9F9", fg="#666")
footer_label.pack(pady=20)

# Run the app
root.mainloop()