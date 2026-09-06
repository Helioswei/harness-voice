import logging
import threading
import time
from collections import deque

import numpy as np
import AVFoundation
import objc
from Foundation import NSRunLoop, NSDate
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger("harness-voice")


class AVRecorder:
    """AVAudioEngine-based recorder with voice processing (echo cancellation).

    Replaces the sounddevice-based AudioRecorder. Uses macOS native
    AVAudioEngine with voiceProcessingEnabled for hardware AEC.
    """

    def __init__(self, samplerate=16000, silence_timeout=1.5,
                 max_record_sec=15):
        self.samplerate = samplerate
        self.silence_timeout = silence_timeout
        self.max_record_sec = max_record_sec

        self._native_sr = None  # set when engine starts (48kHz)

        logger.info("加载 Silero VAD 模型 …")
        self._vad = load_silero_vad(onnx=True)
        logger.info("Silero VAD 加载完毕")

        self._buffer = []
        self._speaking = False
        self._speech_end_time = None
        self._utterance_ready = threading.Event()
        self._utterance_result = None
        self._utterance_completed_at = 0.0
        self._lock = threading.Lock()

        # Wake-word hook
        self._wake_hook = None
        self._wake_triggered = threading.Event()

        self._vad_buffer = np.array([], dtype=np.float32)

        # Ring buffer for trailing audio
        self._ring_buffer = deque(maxlen=int(samplerate * 5))
        self._ring_lock = threading.Lock()

        # Barge-in (TTS 打断)
        self._tts_active = False
        self._tts_vad_threshold = 0.65   # TTS 期间 VAD 阈值
        self._speech_start_time = None   # 当前语音起始时间（用于打断确认）

        # AVAudioEngine
        self._engine = None
        self._running = False
        self._tap_thread = None

    # ------------------------------------------------------------------
    # Public API (same as AudioRecorder)
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._reset_state()

        self._engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = self._engine.inputNode()

        ok, _ = input_node.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            logger.warning("语音处理(AEC)不可用，使用原始麦克风")

        vp_fmt = input_node.outputFormatForBus_(0)
        self._native_sr = int(vp_fmt.sampleRate())
        n_ch = vp_fmt.channelCount()
        logger.info(
            "AVAudioEngine 就绪 (AEC=%s, %d ch, %d Hz)",
            input_node.isVoiceProcessingEnabled(), n_ch, self._native_sr,
        )

        main_mixer = self._engine.mainMixerNode()
        output = self._engine.outputNode()
        self._engine.connect_to_format_(input_node, main_mixer, vp_fmt)
        self._engine.connect_to_format_(main_mixer, output, vp_fmt)
        main_mixer.setOutputVolume_(0.0)

        # Buffer: ~21ms at 48kHz → yields ~341 samples at 16kHz (VAD buffered internally)
        input_node.installTapOnBus_bufferSize_format_block_(
            0, 1024, vp_fmt, self._on_tap,
        )

        self._engine.prepare()
        time.sleep(0.3)
        self._engine.startAndReturnError_(None)

        # Spin a thread to pump the run loop
        self._tap_thread = threading.Thread(
            target=self._run_loop, daemon=True,
        )
        self._tap_thread.start()
        logger.info("音频流已启动 (AEC 已启用)")

    def stop(self):
        self._running = False
        # 解除所有可能的阻塞，防止主线程卡在 wait()
        self._wake_triggered.set()
        self._utterance_ready.set()
        if self._engine:
            try:
                input_node = self._engine.inputNode()
                input_node.removeTapOnBus_(0)
            except Exception:
                pass
            self._engine.stop()
            try:
                self._engine.disconnectNodeInput_(self._engine.outputNode())
            except Exception:
                pass
            self._engine = None

        if self._tap_thread is not None:
            self._tap_thread.join(timeout=2.0)
            self._tap_thread = None

    def close(self):
        self.stop()

    def set_wake_hook(self, hook):
        self._wake_hook = hook
        self._wake_triggered.clear()

    def wait_for_wake_word(self, timeout=None):
        if not self._running:
            self.start()
        self._wake_triggered.clear()
        triggered = self._wake_triggered.wait(timeout=timeout)
        # stop() 会 set() 这个 event 来解除阻塞，此时应返回 False
        return triggered and self._running

    def drain_ring_buffer(self):
        with self._ring_lock:
            if self._ring_buffer:
                audio = np.array(list(self._ring_buffer), dtype=np.float32)
                self._ring_buffer.clear()
                return audio
            return None

    # ── Barge-in support ─────────────────────────────────

    def set_tts_active(self, active, threshold=0.65):
        """Set TTS playback state and VAD threshold during TTS.

        When active, VAD uses a higher threshold to reduce echo residuals
        from triggering false interrupts. The VAD state machine continues
        uninterrupted — in-progress buffers are preserved.

        When entering TTS mode, stale ``_utterance_result`` from prior
        VAD activity (e.g. background music) is cleared so only speech
        captured *during* this TTS can trigger a barge-in.
        """
        with self._lock:
            self._tts_active = active
            self._tts_vad_threshold = threshold
            if active:
                self._utterance_result = None
                self._utterance_completed_at = 0.0
                self._utterance_ready.clear()
                logger.info("TTS-active=True (VAD threshold=%.2f)", threshold)
            else:
                self._speech_start_time = None
                logger.info("TTS-active=False")

    def check_interrupt(self, min_duration=0.3):
        """Check if user has been speaking continuously for *min_duration* during TTS.

        Only the ``_speaking`` path is used — ``_utterance_result`` is deliberately
        excluded because coughs/throat-clearing trigger VAD completion and would
        cause false barge-ins that bypass the duration gate.

        Speech captured during HTTP wait is handled by ``_renew_if_pending()``
        in ``main.py``, which consumes it **before** TTS ever starts.
        """
        with self._lock:
            if (self._speaking and self._speech_start_time is not None
                    and self._speech_end_time is None):
                elapsed = time.monotonic() - self._speech_start_time
                return elapsed >= min_duration
            return False

    def consume_utterance(self):
        """Atomically retrieve a completed utterance without clearing VAD state.

        Use this after an interrupt to grab the audio the user spoke during
        TTS playback.
        """
        with self._lock:
            if self._utterance_result is not None:
                audio = self._utterance_result
                self._utterance_result = None
                self._utterance_ready.clear()
                return audio
            return None

    def wait_utterance(self, timeout=3.0):
        """Wait for VAD to complete an utterance, preserving in-progress buffer.

        Unlike ``read_utterance()``, this does **not** reset VAD state or
        clear the audio buffer — any speech already captured when barge-in
        fired is preserved.
        """
        if self._utterance_ready.wait(timeout=timeout):
            with self._lock:
                audio = self._utterance_result
                self._utterance_result = None
                self._utterance_ready.clear()
                return audio
        return None

    def read_utterance(self, idle_timeout=None):
        if not self._running:
            self.start()

        self._utterance_ready.clear()
        self._utterance_result = None
        self._reset_state()

        timeout_at = None
        if idle_timeout is not None:
            timeout_at = time.monotonic() + idle_timeout

        while self._running and self._engine is not None:
            remaining = _remaining_or_none(timeout_at)
            if remaining is not None and remaining <= 0:
                self._reset_state()
                return None

            if self._utterance_ready.wait(timeout=min(remaining or 0.1, 0.1)):
                with self._lock:
                    audio = self._utterance_result
                    self._utterance_result = None
                    self._utterance_ready.clear()
                if audio is not None:
                    return audio

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        while self._running:
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.05)
            )

    def _reset_state(self):
        with self._lock:
            self._buffer = []
            self._speaking = False
            self._speech_start_time = None
            self._speech_end_time = None
            self._vad.reset_states()
        self._vad_buffer = np.array([], dtype=np.float32)

    def _on_tap(self, buf, when):
        """AVAudioEngine tap callback — runs on audio thread."""
        if not self._running:
            return

        n_frames = int(buf.frameLength())
        ch_data = buf.floatChannelData()
        if ch_data is None:
            return

        # Channel 0 = AEC-processed mic signal
        ch0 = ch_data[0]
        raw = ch0.as_buffer(n_frames)
        audio_48k = np.frombuffer(raw, dtype=np.float32)

        # Decimate 48kHz → 16kHz
        step = self._native_sr // self.samplerate
        mono = audio_48k[::step].copy()

        # ── Wake-word hook (KWS) ─────────────────────
        if self._wake_hook is not None:
            try:
                int16_chunk = (mono * 32767).astype("int16")
                if self._wake_hook(int16_chunk):
                    self._wake_triggered.set()
            except Exception as exc:
                logger.error("唤醒检测失败: %s", exc)

        # ── Ring buffer ──────────────────────────────
        with self._ring_lock:
            self._ring_buffer.extend(mono.tolist())

        # ── VAD (process 512-sample chunks) ─────────
        self._vad_buffer = np.concatenate([self._vad_buffer, mono])

        while len(self._vad_buffer) >= 512:
            chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]

            try:
                tensor = torch.from_numpy(chunk)
                probs = self._vad.audio_forward(tensor, self.samplerate)
                prob = probs.flatten()[0].item()
            except Exception as exc:
                logger.error("VAD 失败: %s", exc)
                continue

            now = time.monotonic()
            with self._lock:
                vad_threshold = self._tts_vad_threshold if self._tts_active else 0.5
                if prob > vad_threshold:
                    if not self._speaking:
                        logger.debug("VAD 检测到语音 (prob=%.3f)", prob)
                        self._speaking = True
                        self._speech_start_time = now  # 记录语音起始时间
                        self._buffer = []
                        self._speech_end_time = None
                    self._speech_end_time = None
                    self._buffer.append(chunk)
                elif self._speaking:
                    self._buffer.append(chunk)
                    if self._speech_end_time is None:
                        self._speech_end_time = now
                    elif now - self._speech_end_time >= self.silence_timeout:
                        logger.debug("语音结束 (%.1fs 静音)", self.silence_timeout)
                        audio = np.concatenate(self._buffer)
                        self._utterance_result = audio
                        self._utterance_completed_at = now
                        self._speaking = False
                        self._buffer = []
                        self._speech_end_time = None
                        self._speech_start_time = None
                        self._utterance_ready.set()
                        self._vad.reset_states()


def _remaining_or_none(timeout_at):
    if timeout_at is None:
        return None
    return timeout_at - time.monotonic()
