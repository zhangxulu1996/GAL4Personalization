import clip
import torch
from transformers import AutoImageProcessor
from transformers import AutoImageProcessor, AutoModel

class DINOEvaluator(object):
    def __init__(self, device, dino_model='facebook/dinov2-base'):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(dino_model)
        self.model = AutoModel.from_pretrained(dino_model).to(device)

    @torch.no_grad()
    def encode_images(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)
        
        return image_features

    def get_image_features(self, img, norm=True) -> torch.Tensor:
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def img_to_img_similarity(self, generated_images, src_images=None, src_img_features=None):
        if src_img_features is None and src_images is not None:
            src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()




class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)
        
        self.preprocess = clip_preprocess

    def tokenize(self, strings):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images):
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text, norm = True):

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img, norm=True):
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, generated_images, src_images=None, src_img_features=None):
        if src_img_features is None and src_images is not None:
            src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, generated_images, text=None, text_features=None):
        if text_features is None and text is not None:
            text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()
