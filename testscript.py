from PIL import Image
import torch
import torchvision.transforms as T
import time

def testscript():
    # Set label map for NTZ dataset
    label_map = {0: "fail_label_crooked_print",
                 1: "fail_label_half_printed",
                 2: "fail_label_not_fully_printed",
                 3: "no_fail"}

    # Set NTZ transforms
    NTZ_transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Get model
    model = torch.load("model.pth", map_location = torch.device(device))
    model.eval()

    # There should be a for loop here that checks if a new image has been sent to location
    # If there is a new image, a prediction should be made. If there is no new image
    # The check should go again (maybe after a certain time interval? like 1 second?)
    input = Image.open("test_image.png")
    input = NTZ_transform(input)
    with torch.no_grad():
        input = input.to(device)
        model_output = model(input)
        label = model_output.argmax(dim = 1)

    print("Model class prediction: " + label_map[label.item()])
    # Example sleep call = time.sleep(1)

if __name__ == '__main__':
    testscript()