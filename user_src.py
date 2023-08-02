import torch
from diffusers import StableDiffusionDepth2ImgPipeline
import base64
from io import BytesIO
from PIL import Image
import requests
from typing import Callable, Dict, List, Union

def extract_token_from_dict(embedding_dict: Dict[str, str]) -> str:
        r"""
        Extracts the token from the embedding dictionary.

        Arguments:
            embedding_dict (`Dict[str, str]`):
                The embedding dictionary loaded from the embedding path

        Returns:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
        """
        # auto1111 embedding case
        if "string_to_param" in embedding_dict:
            token = embedding_dict["name"]
            return token

        return list(embedding_dict.keys())[0]

def extract_embedding_from_dict(embedding_dict: Dict[str, str]) -> torch.Tensor:
    r"""
    Extracts the embedding from the embedding dictionary.
    Arguments:
        embedding_dict (`Dict[str, str]`):
            The embedding dictionary loaded from the embedding path
    Returns:
        embedding (`torch.Tensor`):
            The embedding to be added to the text encoder's embedding matrix
    """
    # auto1111 embedding case
    if "string_to_param" in embedding_dict:
        embedding_dict = embedding_dict["string_to_param"]
        embedding = embedding_dict["*"]
        return embedding
    return list(embedding_dict.values())[0]

def add_textual_inversion_embedding(model, token: str, embedding: torch.Tensor):
        r"""
        Adds a token to the tokenizer's vocabulary and an embedding to the text encoder's embedding matrix.

        Arguments:
            token (`str`):
                The token to be added to the tokenizers' vocabulary
            embedding (`torch.Tensor`):
                The embedding of the token to be added to the text encoder's embedding matrix
        """
    

        embedding = embedding.to(model.text_encoder.dtype)

        if token in model.tokenizer.get_vocab():
            # If user has allowed replacement and the token exists, we only need to
            # extract the existing id and update the embedding
            token_id = model.tokenizer.convert_tokens_to_ids(token)
            model.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            return model
        else:
            # If the token does not exist, we add it to the tokenizer, then resize and update the
            # text encoder acccordingly
            model.tokenizer.add_tokens([token])

            token_id = model.tokenizer.convert_tokens_to_ids(token)
            # NOTE: len() does't start at 0, so we shouldn't need to +1
            # since we already updated the tokenizer and it's new length
            # should be old length + 1
            model.text_encoder.resize_token_embeddings(len(model.tokenizer))
            model.text_encoder.get_input_embeddings().weight.data[token_id] = embedding
            return model

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    model = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth",torch_dtype=torch.float16).to("cuda")
    
    textual_inversion_embedding = torch.load("emb.pt",map_location="cuda")
    token = extract_token_from_dict(textual_inversion_embedding)
    embedding = extract_embedding_from_dict(textual_inversion_embedding)
    embedding = embedding[0]
    model = add_textual_inversion_embedding(model,token,embedding)
   

    return {'model': model}

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model, model_inputs) -> dict:
    # Parse out your arguments
    image_url = model_inputs.get('imageURL', None)
    if image_url == None:
        return {'message': "No image url provided"}
    
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    n_prompt = model_inputs.get('negative_prompt', "")
    strength = model_inputs.get('strength', 0.8)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)


    print(model_inputs.get('imageURL', None))
    print(model_inputs.get('strength', 0.7))
    print(model_inputs.get('prompt', None))
    print(model_inputs.get('negative_prompt', ""))

    init_image = Image.open(requests.get(image_url, stream=True).raw)

#   output_image = pipe(prompt=prompt, image=init_image).images[0]
#   output_images = model(prompt=prompt, image=init_image, num_images_per_prompt=4)

    output_images = model(prompt=prompt, image=init_image, num_images_per_prompt=4, negative_prompt=n_prompt, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
 
    output_image = output_images[0]

    buffered = BytesIO()
    output_image.save(buffered,format="JPEG")
    image_base64_0 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    output_image = output_images[1]

    buffered = BytesIO()
    output_image.save(buffered,format="JPEG")
    image_base64_1 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    output_image = output_images[2]

    buffered = BytesIO()
    output_image.save(buffered,format="JPEG")
    image_base64_2 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    output_image = output_images[3]
    
    buffered = BytesIO()
    output_image.save(buffered,format="JPEG")
    image_base64_3 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        
    # Return the results as a dictionary
    return {'image_base64_0': image_base64_0, 'image_base6_1': image_base64_1, 'image_base64_2': image_base64_2, 'image_base64_3': image_base64_3}
