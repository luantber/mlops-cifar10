import gradio as gr
from model.predict import predict
from torchvision.transforms import functional as F

def custom_predict(image):
    image = F.to_tensor(image)
    return predict(image, get_dictionary=True)



demo = gr.Interface(
    custom_predict,
    title="Image Classifier using CNN ( Cifar-10) ",
    description="This is a image classifier using a CNN, it was trained on the Cifar-10 dataset ( Kaggle) \n",
    article="The architecture is a CNN, uploaded via Github Actions",
    inputs=gr.Image(shape=(32, 32),type="pil"),
    outputs=gr.Label(),
    examples=["examples/1.png", "examples/2.png", "examples/3.png", "examples/4.png" , "examples/5.png"],
)

demo.launch()