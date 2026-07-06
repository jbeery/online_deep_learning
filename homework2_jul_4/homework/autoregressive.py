import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens

        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.start_embedding = torch.nn.Parameter(torch.zeros(1, 1, d_latent))
        self.position_embedding = torch.nn.Embedding(1024, d_latent)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=4,
            dim_feedforward=4 * d_latent,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=2,
        )

        self.output = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape

        x = x.reshape(B, h * w)
        x = self.token_embedding(x)

        x = torch.cat(
            [
                self.start_embedding.expand(B, 1, -1),
                x[:, :-1],
            ],
            dim=1,
        )

        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)[None, :, :]

        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.shape[1],
            device=x.device,
        )

        x = self.transformer(x, mask=mask)
        x = self.output(x)

        return x.reshape(B, h, w, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()

        tokens = torch.zeros(
            B,
            h * w,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            for i in range(h * w):
                logits, _ = self.forward(tokens.reshape(B, h, w))
                logits = logits.reshape(B, h * w, self.n_tokens)[:, i]
                tokens[:, i] = torch.distributions.Categorical(
                    logits=logits
                ).sample()

        if was_training:
            self.train()

        return tokens.reshape(B, h, w)