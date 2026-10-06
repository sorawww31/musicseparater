"""backend/model_runtime/singer_informed.py: 公開重みと一致する条件付きOpen-Unmixを定義する。"""

from __future__ import annotations

from typing import Any


def build_models(settings: dict[str, Any], torch: Any) -> tuple[Any, Any]:
    """PyTorchを子プロセス内だけで受け取り、separatorとembedding modelを構築する。"""

    class SingerEmbedding(torch.nn.Module):
        """3秒mono音声を32次元のsinger embeddingへ変換する2層GRU。"""

        def __init__(self) -> None:
            super().__init__()
            self.n_fft = settings["embedding_n_fft"]
            self.hop_length = settings["embedding_hop_length"]
            self.register_buffer(
                "window",
                torch.hann_window(self.n_fft),
                persistent=False,
            )
            self.rnn = torch.nn.GRU(
                input_size=self.n_fft // 2 + 1,
                hidden_size=settings["embedding_size"],
                num_layers=settings["embedding_layers"],
                batch_first=True,
            )

        def embedding(self, waveform: Any) -> Any:
            spectrum = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
            )
            magnitude = torch.abs(spectrum).permute(0, 2, 1)
            recurrent, _ = self.rnn(magnitude)
            return recurrent[:, -1]

    class SingerInformedSeparator(torch.nn.Module):
        """spectrumへembeddingを連結する論文のOpen-Unmixモデル。"""

        def __init__(self) -> None:
            super().__init__()
            self.output_bins = settings["n_fft"] // 2 + 1
            self.max_bin = settings["max_bin"]
            self.channels = settings["channels"]
            self.hidden_size = settings["hidden_size"]
            embedding_size = settings["embedding_size"]

            self.fc1 = torch.nn.Linear(
                self.max_bin * self.channels + embedding_size,
                self.hidden_size,
                bias=False,
            )
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_size)
            self.lstm = torch.nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size // 2,
                num_layers=settings["lstm_layers"],
                bidirectional=True,
                batch_first=False,
                dropout=settings["lstm_dropout"],
            )
            self.fc2 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
            self.bn2 = torch.nn.BatchNorm1d(self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, self.output_bins * self.channels, bias=False)
            self.bn3 = torch.nn.BatchNorm1d(self.output_bins * self.channels)

            self.input_mean = torch.nn.Parameter(torch.zeros(self.max_bin))
            self.input_scale = torch.nn.Parameter(torch.ones(self.max_bin))
            self.output_scale = torch.nn.Parameter(torch.ones(self.output_bins))
            self.output_mean = torch.nn.Parameter(torch.ones(self.output_bins))

        def forward(self, magnitude: Any, embedding: Any) -> Any:
            features = magnitude.permute(3, 0, 1, 2)
            frame_count, batch_size, channels, _ = features.shape
            mixture = features.detach().clone()

            features = features[..., : self.max_bin]
            features = (features + self.input_mean) * self.input_scale
            features = features.reshape(frame_count, batch_size, channels * self.max_bin)
            condition = embedding.reshape(batch_size, -1).unsqueeze(0)
            condition = condition.expand(frame_count, -1, -1)
            features = torch.cat((features, condition), dim=-1)

            features = self.fc1(features.reshape(-1, features.shape[-1]))
            features = self.bn1(features)
            features = torch.tanh(features.reshape(frame_count, batch_size, self.hidden_size))
            recurrent, _ = self.lstm(features)
            features = torch.cat((features, recurrent), dim=-1)

            features = self.fc2(features.reshape(-1, features.shape[-1]))
            features = torch.nn.functional.relu(self.bn2(features))
            features = self.bn3(self.fc3(features))
            features = features.reshape(
                frame_count,
                batch_size,
                channels,
                self.output_bins,
            )
            features = torch.nn.functional.relu(
                features * self.output_scale + self.output_mean
            )
            return (features * mixture).permute(1, 2, 3, 0)

    return SingerInformedSeparator(), SingerEmbedding()
