import logging
import os
import glob
import shutil

import numpy as np
import sherpa_onnx
from pypinyin import pinyin, Style

from .models import validate_model_files

logger = logging.getLogger("harness-voice")


def _generate_keywords_line(text):
    """Convert Chinese text to sherpa-onnx ppinyin keyword format.

    Example: "小九" → "x iǎo j iǔ @小九"
    """
    initials = pinyin(text, style=Style.INITIALS, strict=False)
    finals = pinyin(text, style=Style.FINALS_TONE, strict=False)
    tokens = []
    for i, f in zip(initials, finals):
        tokens.append(i[0])
        tokens.append(f[0])
    return " ".join(tokens) + f" @{text}"


class WakeWordEngine:
    """Sherpa-ONNX keyword spotter for wake word detection.

    Model files are stored in ``models/kws/`` under the project root.
    Downloaded automatically from ModelScope on first use.

    Parameters
    ----------
    keywords : list[str]
        Wake words to detect, e.g. ``["小九"]``.
    threshold : float
        Detection threshold (0.0–1.0). Higher = fewer false triggers.
    """

    def __init__(self, keywords=None, threshold=0.25, keywords_score=None):
        if keywords is None:
            keywords = ["小九"]

        model_dir = self._resolve_model_dir()
        keywords_path = os.path.join(model_dir, "hermes_keywords.txt")
        self._write_keywords(keywords, keywords_path)

        tokens_path = os.path.join(model_dir, "tokens.txt")
        encoder = self._find(model_dir, "encoder-*.onnx")
        decoder = self._find(model_dir, "decoder-*.onnx")
        joiner = self._find(model_dir, "joiner-*.onnx")

        validate_model_files(model_dir, [
            "tokens.txt",
            os.path.basename(encoder),
            os.path.basename(decoder),
            os.path.basename(joiner),
        ])

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens_path,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_path,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            keywords_threshold=threshold,
        )
        self._stream = self._spotter.create_stream()
        self._last_keyword = None
        logger.info(
            "唤醒引擎就绪 (关键词=%s, 阈值=%.2f)",
            keywords, threshold,
        )

    # ------------------------------------------------------------------
    def process_chunk(self, audio: np.ndarray):
        """Feed a chunk of int16 audio samples.

        Returns the detected keyword string, or None.
        """
        self._stream.accept_waveform(16000, audio)
        if self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                logger.debug("唤醒词检出: %s", result)
                self._last_keyword = result
                return result
        return None

    @property
    def last_keyword(self):
        return self._last_keyword

    # ------------------------------------------------------------------
    def reset(self):
        """Reset the internal stream (call after a keyword is detected)."""
        self._last_keyword = None
        self._stream = self._spotter.create_stream()

    # ------------------------------------------------------------------
    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_dir():
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        model_dir = os.path.join(project_root, "models", "kws")
        os.makedirs(model_dir, exist_ok=True)

        if os.path.isfile(os.path.join(model_dir, "tokens.txt")):
            return model_dir

        logger.info("正在从 ModelScope 下载唤醒模型 …")
        from modelscope import snapshot_download

        raw = snapshot_download(
            "pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        )
        for f in os.listdir(raw):
            src = os.path.join(raw, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(model_dir, f))

        logger.info("唤醒模型已下载至 %s", model_dir)
        return model_dir

    @staticmethod
    def _find(directory, pattern):
        files = glob.glob(os.path.join(directory, pattern))
        if not files:
            raise FileNotFoundError(
                f"No file matching {pattern} in {directory}"
            )
        return files[0]

    @staticmethod
    def _write_keywords(keywords, path):
        with open(path, "w") as f:
            for kw in keywords:
                f.write(_generate_keywords_line(kw) + "\n")
