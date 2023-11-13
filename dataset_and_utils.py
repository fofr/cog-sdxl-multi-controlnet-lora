from typing import List, Optional

import torch
import torch.utils.checkpoint
from safetensors import safe_open


class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    def _load_embeddings(self, loaded_embeddings, tokenizer, text_encoder):
        # Assuming new tokens are of the format <s_i>
        self.inserting_toks = [f"<s{i}>" for i in range(loaded_embeddings.shape[0])]
        special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
        tokenizer.add_special_tokens(special_tokens_dict)
        text_encoder.resize_token_embeddings(len(tokenizer))

        self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
        assert self.train_ids is not None, "New tokens could not be converted to IDs."
        text_encoder.text_model.embeddings.token_embedding.weight.data[
            self.train_ids
        ] = loaded_embeddings.to(device=self.device).to(dtype=self.dtype)

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                index_no_updates
            ] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = (
                text_encoder.text_model.embeddings.token_embedding.weight.data[
                    index_updates
                ]
            )
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            text_encoder.text_model.embeddings.token_embedding.weight.data[
                index_updates
            ] = new_embeddings

    def load_embeddings(self, file_path: str):
        with safe_open(file_path, framework="pt", device=self.device.type) as f:
            for idx in range(len(self.text_encoders)):
                text_encoder = self.text_encoders[idx]
                tokenizer = self.tokenizers[idx]

                loaded_embeddings = f.get_tensor(f"text_encoders_{idx}")
                self._load_embeddings(loaded_embeddings, tokenizer, text_encoder)
