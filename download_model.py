import urllib.request

url = "https://huggingface.co/DikshitaP/student-placement-model/resolve/main/placement_model_best.pt"

urllib.request.urlretrieve(url, "placement_model_best.pt")

print("Model downloaded successfully!")