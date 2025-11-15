# Based on https://github.com/alessandroragano/scoreq
# Modified to run on tensors
import math
import os
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# The wav2vec 2.0 model's CNN feature extractor has a total stride of 320
PADDING_MULTIPLE = 320


def dynamic_pad(x, multiple=PADDING_MULTIPLE, dim=-1, value=0):
    """Pads the input tensor to be a multiple of PADDING_MULTIPLE."""
    tsz = x.size(dim)
    required_len = math.ceil(tsz / multiple) * multiple
    remainder = required_len - tsz
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, pad_offset + (0, remainder), value=value)


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(n - self.n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# PyTorch classes needed for the use_onnx=False fallback
class TripletModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(nn.ReLU(), nn.Linear(self.ssl_features, emb_dim))

    def forward(self, wav, phead=False):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        if phead:
            x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class MosPredictor(nn.Module):
    def __init__(self, pt_model, emb_dim=768):
        super(MosPredictor, self).__init__()
        self.pt_model = pt_model
        self.mos_layer = nn.Linear(emb_dim, 1)

    def forward(self, wav):
        x = self.pt_model(wav, phead=False)
        if len(x.shape) == 3: x.squeeze_(2)
        out = self.mos_layer(x)
        return out


class Scoreq(nn.Module):
    """
    Main class for handling the SCOREQ audio quality assessment model.
    Defaults to using high-performance ONNX models.
    """

    def __init__(self, data_domain='natural', mode='nr'):
        """
        Initializes the Scoreq object.

        Args:
            data_domain (str): Domain of audio ('natural' or 'synthetic').
            mode (str): Mode of operation ('nr' or 'ref').
        """
        super().__init__()
        self.data_domain = data_domain
        self.mode = mode
        self.model = None
        self.session = None
        self._init_pytorch()

    def _init_pytorch(self):
        """Initializes the original PyTorch/fairseq model."""
        try:
            import fairseq
        except ImportError:
            raise ImportError(
                "PyTorch/fairseq mode requires 'fairseq' and 'torch'. "
                "Please install them with: pip install scoreq[pytorch]"
            )

        url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        CHECKPOINT_PATH = self._download_model("wav2vec_small.pt", url_w2v, "pt-models")

        # Temporarily monkey-patch torch.load to default to weights_only=False.
        # This is necessary because fairseq's internal loading function does not
        # expose this argument, and it's required for newer PyTorch versions to
        # load old checkpoints containing non-tensor data.
        original_torch_load = torch.load
        try:
            def new_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)

            torch.load = new_torch_load

            w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
        finally:
            torch.load = original_torch_load

        ssl_model = w2v_model[0]
        ssl_model.remove_pretraining_modules()

        pt_model = TripletModel(ssl_model, ssl_out_dim=768, emb_dim=256)

        if self.mode == 'nr':
            model = MosPredictor(pt_model, emb_dim=768)
        else:
            model = pt_model

        PT_URLS = {
            ('natural', 'nr'): 'https://zenodo.org/records/13860326/files/adapt_nr_telephone.pt',
            ('natural', 'ref'): 'https://zenodo.org/records/13860326/files/fixed_nmr_telephone.pt',
            ('synthetic', 'nr'): 'https://zenodo.org/records/13860326/files/adapt_nr_synthetic.pt',
            ('synthetic', 'ref'): 'https://zenodo.org/records/13860326/files/fixed_nmr_synthetic.pt',
        }
        model_key = (self.data_domain, self.mode)
        model_url = PT_URLS.get(model_key)
        if not model_url:
            raise ValueError(f"Invalid model combination: domain='{self.data_domain}', mode='{self.mode}'")

        MODEL_PATH = self._download_model(os.path.basename(model_url), model_url, "pt-models")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))

        self.model = model
        self.model.eval()

    def _download_model(self, filename, url, cache_dir_name):
        """Helper to download a model from a URL with a progress bar."""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "scoreq", cache_dir_name)
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, filename)

        if not os.path.exists(model_path):
            print(f"Downloading {filename}...")
            try:
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urlretrieve(url, model_path, reporthook=t.update_to)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                if os.path.exists(model_path): os.remove(model_path)
                raise e

        return model_path

    def preprocess_tensor(self, wave, sr):
        """Preprocesses an audio tensor for prediction."""
        if wave.shape[0] > 1: wave = wave.mean(dim=0, keepdim=True)
        if sr != 16000: wave = torchaudio.transforms.Resample(sr, 16000)(wave)
        wave = dynamic_pad(wave, PADDING_MULTIPLE)
        return wave

    def forward(self, inf, fs, ref=None, **kwargs):
        """Forward method to make the class callable."""
        if self.mode == 'nr':
            return self.predict_tensor(inf, fs)
        else:
            return self.predict_tensor(inf, fs, ref, fs)

    @torch.inference_mode()
    def predict_tensor(self, test_wave, test_sr, ref_wave=None, ref_sr=None):
        """Makes predictions on audio tensors."""
        test_wave_padded = self.preprocess_tensor(test_wave, test_sr)

        with torch.no_grad():
            if self.mode == 'nr':
                score = self.model(test_wave_padded)
            else:
                if ref_wave is None: raise ValueError("ref_path must be provided.")
                ref_wave_padded = self.preprocess_tensor(ref_wave, ref_sr)
                test_emb = self.model(test_wave_padded)
                ref_emb = self.model(ref_wave_padded)
                score = torch.cdist(test_emb, ref_emb)
        return score.item() if score.numel() == 1 else score

    def predict(self, test_path, ref_path=None):
        """Makes predictions on audio files."""
        return self._predict_pytorch(test_path, ref_path)

    @property
    def device(self):
        """Returns the device on which the model is loaded."""
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                print("Unable to determine device from model parameters, defaulting to 'cpu'.")
        return 'cpu'

    def _predict_pytorch(self, test_path, ref_path=None):
        """Prediction using the original PyTorch model."""
        test_wave_raw = self.load_processing(test_path)
        test_wave_padded = dynamic_pad(test_wave_raw).to(self.device)

        with torch.no_grad():
            if self.mode == 'nr':
                score = self.model(test_wave_padded).item()
            else:
                if ref_path is None: raise ValueError("ref_path must be provided.")
                ref_wave_raw = self.load_processing(ref_path)
                ref_wave_padded = dynamic_pad(ref_wave_raw).to(self.device)

                test_emb = self.model(test_wave_padded)
                ref_emb = self.model(ref_wave_padded)
                score = torch.cdist(test_emb, ref_emb).item()
        return score

    def load_processing(self, filepath, target_sr=16000):
        """Loads and preprocesses an audio file."""
        wave, sr = torchaudio.load(filepath)
        if wave.shape[0] > 1: wave = wave.mean(dim=0, keepdim=True)
        if sr != target_sr: wave = torchaudio.transforms.Resample(sr, target_sr)(wave)
        return wave