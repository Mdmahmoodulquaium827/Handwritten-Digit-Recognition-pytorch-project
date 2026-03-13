import torch
import torch.nn as nn
import torch.nn.functional as F
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from torchvision import transforms







class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



device = torch.device("cuda")

model = CNN()
model.load_state_dict(torch.load("I:\Python\PyTorch Project Handwritten Digit Recognition\mnist_cnn.pth", map_location=device))
model.eval()



'''transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)  
])'''
transform = transforms.ToTensor()  



def predict_digit(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    return prediction



ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.geometry("400x350")
app.title("Handwritten Digit Recognition")


def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if file_path:
        digit = predict_digit(file_path)
        result_label.configure(text=f"Predicted Digit: {digit}")


title_label = ctk.CTkLabel(
    app,
    text="Handwritten Digit Recognition",
    font=("Arial", 18)
)
title_label.pack(pady=20)


upload_btn = ctk.CTkButton(
    app,
    text="Upload Digit Image",
    command=upload_image
)
upload_btn.pack(pady=20)


result_label = ctk.CTkLabel(
    app,
    text="Predicted Digit: ",
    font=("Arial", 22)
)
result_label.pack(pady=30)


app.mainloop()
