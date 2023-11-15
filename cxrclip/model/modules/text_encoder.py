from torch import nn
from transformers import AutoConfig, AutoModel, BertModel


class HuggingfaceTextEncoder(nn.Module):
    def __init__(
        self,
        name: str = "bert-base-uncased",
        vocab_size: int = None,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = "~/.cache/huggingface/hub",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        if pretrained:
            self.text_encoder = AutoModel.from_pretrained(
                name,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            # initializing with a config file does not load the weights associated with the model
            model_config = AutoConfig.from_pretrained(
                name,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            if type(model_config).__name__ == "BertConfig":
                self.text_encoder = BertModel(model_config)
            else:
                # TODO: add text models if needed
                raise NotImplementedError(f"Not support training from scratch : {type(model_config).__name__}")

        if gradient_checkpointing and self.text_encoder.supports_gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()

        self.out_dim = self.text_encoder.config.hidden_size

    def forward(self, x):
        output = self.text_encoder(**x)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)
