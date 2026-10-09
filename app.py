import json
import os
import torch
import torch.nn as nn
import gradio as gr
from torchvision import models, transforms

with open("classes.json") as f:
    classes = json.load(f)

tf = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = models.resnet18()
model.fc = nn.Linear(512, len(classes))
model.load_state_dict(torch.load("resnet18_frozen.pt", map_location="cpu"))
model.eval()

LOW_CONF = 0.5
LOW_CONF_NOTE = (
    "**Low confidence:** the model is unsure. "
    "The breed may be wrong, or may not be one of the 37 it knows."
)


def predict(img):
    if img is None:
        return {}, ""
    x = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    result = {classes[i]: float(p) for p, i in zip(top.values, top.indices)}
    note = LOW_CONF_NOTE if float(top.values[0]) < LOW_CONF else ""
    return result, note


breed_list = ", ".join(str(c).replace("_", " ") for c in classes)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a cat or dog photo"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top 3 breeds"),
        gr.Markdown(),
    ],
    flagging_mode="never",
    title="Pet Breed Classifier (ResNet18, 37 breeds)",
    description=(
        "Pretrained ResNet18 with a trained linear head, on Oxford-IIIT Pet. "
        "84.1% test accuracy in the original run (3-seed mean 84.2%). "
        "Look-alike breeds, such as Pit Bull and Staffordshire Bull Terrier, are often confused. "
        "The model has no 'not a pet' option, so any image gets one of the 37 breeds. "
        "Code and evaluation: https://github.com/dmatharv25-cmd/pet-breed-classifier"
    ),
    article="**Breeds this model knows (" + str(len(classes)) + "):** " + breed_list,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
