import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class FaceRecognition:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = "./dinov2-base"
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        self.embedder = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True
        ).to(self.device)

        self.embedder.eval()

    @torch.no_grad()
    def get_embedding(self, face_img: np.ndarray):
        """
        face_img: np.ndarray (H, W, 3), RGB, 얼굴 crop된 이미지
        return: np.ndarray (768,)
        """
        if face_img.dtype != np.uint8:
            face_img = face_img.astype(np.uint8)

        face_pil = Image.fromarray(face_img).convert("RGB")

        inputs = self.processor(
            images=face_pil,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.embedder(**inputs)

        # last_hidden_state: (1, tokens, 768)
        feats = outputs.last_hidden_state

        # CLS token
        emb = feats[:, 0]                  # (1, 768)
        emb = F.normalize(emb, p=2, dim=1) # L2 normalize

        return emb.squeeze(0).cpu().numpy()

    @staticmethod
    def compute_similarity_distance(query_face_embedding, registered_face_embedding):
        """
        Cosine distance
        return 값이 작을수록 같은 사람
        """
        q = np.asarray(query_face_embedding)
        r = np.asarray(registered_face_embedding)

        sim = np.dot(q, r) / (np.linalg.norm(q) * np.linalg.norm(r))
        distance = 1.0 - sim
        return distance