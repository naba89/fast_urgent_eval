import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from espnet2.bin.spk_inference import Speech2Embedding
from transformers import WavLMForXVector
from huggingface_hub import hf_hub_download


class SpeakerSimilarity(nn.Module):
    def __init__(self, cosine_similarity=True):
        super().__init__()
        self.spk_model = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_rawnet3").spk_model
        self.spk_model = self.spk_model.eval()
        self.sample_rate = 16000
        self.cosine_similarity = cosine_similarity
        for param in self.spk_model.parameters():
            param.requires_grad = False

    def forward(self, ref, inf, fs):
        """
        inf: torch.Tensor, shape (B, T)
        ref: torch.Tensor, shape (B, T)
        fs: int
        """

        # Resample the input to 16kHz
        inf = torchaudio.functional.resample(inf, fs, self.sample_rate)
        ref = torchaudio.functional.resample(ref, fs, self.sample_rate)

        inf_inputs = {"speech": inf, "extract_embd": True}
        ref_inputs = {"speech": ref, "extract_embd": True}

        # b. Forward the model embedding extraction
        inf_emb = self.spk_model(**inf_inputs)
        ref_emb = self.spk_model(**ref_inputs)

        if self.cosine_similarity:
            # Normalize the embeddings
            inf_emb = F.normalize(inf_emb, p=2, dim=-1)
            ref_emb = F.normalize(ref_emb, p=2, dim=-1)

            similarity = torch.cosine_similarity(inf_emb, ref_emb, dim=-1)
        else:
            similarity = F.l1_loss(inf_emb, ref_emb)

        similarity = similarity.mean()
        return similarity.item()


class SpeakerSimilarityWavLM(nn.Module):
    def __init__(self, cosine_similarity=True):
        super().__init__()
        self.spk_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        for param in self.spk_model.parameters():
            param.requires_grad = False

        self.cosine_similarity = cosine_similarity
        self.sample_rate = 16000

    def forward(self, ref, inf, fs):
        """
        inf: torch.Tensor, shape (B, T)
        ref: torch.Tensor, shape (B, T)
        fs: int
        """
        # Resample the input to 16kHz

        inf = torchaudio.functional.resample(inf, fs, self.sample_rate)
        ref = torchaudio.functional.resample(ref, fs, self.sample_rate)

        inf_emb = self.spk_model(inf).embeddings
        ref_emb = self.spk_model(ref).embeddings

        if self.cosine_similarity:
            # Normalize the embeddings
            inf_emb = F.normalize(inf_emb, dim=-1)
            ref_emb = F.normalize(ref_emb, dim=-1)

            similarity = torch.cosine_similarity(inf_emb, ref_emb, dim=-1)
        else:
            similarity = F.l1_loss(inf_emb, ref_emb)

        similarity = similarity.mean()
        return similarity.item()


class SpeakerSimilarityEcapa2(nn.Module):
    def __init__(self, cosine_similarity=True):
        super().__init__()
        model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)

        self.spk_model = torch.jit.load(model_file, map_location='cpu').eval()
        for param in self.spk_model.parameters():
            param.requires_grad = False

        self.cosine_similarity = cosine_similarity
        self.sample_rate = 16000

    def forward(self, ref, inf, fs, **kwargs):
        """
        inf: torch.Tensor, shape (B, T)
        ref: torch.Tensor, shape (B, T)
        """
        # Resample the input to 16kHz
        inf = torchaudio.functional.resample(inf, fs, self.sample_rate)
        ref = torchaudio.functional.resample(ref, fs, self.sample_rate)

        inf_emb = self.spk_model(inf)
        ref_emb = self.spk_model(ref)

        if self.cosine_similarity:
            # Normalize the embeddings
            inf_emb = F.normalize(inf_emb, dim=-1)
            ref_emb = F.normalize(ref_emb, dim=-1)

            similarity = torch.cosine_similarity(inf_emb, ref_emb, dim=-1)
        else:
            similarity = F.l1_loss(inf_emb, ref_emb)

        similarity = similarity.mean()
        return similarity
