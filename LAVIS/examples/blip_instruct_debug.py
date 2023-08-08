
#%%
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

#%%

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   




# setup device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

#%%
# we associate a model with its preprocessors to make it easier for inference.
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
)



#%%
from rich.console import Console


print(vis_processors.keys())

#%%
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


model.generate({"image": image})

