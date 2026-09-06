import logging
import os
import signal
import subprocess
import sys
import time
import warnings
from logging.handlers import RotatingFileHandler

import AVFoundation
import yaml

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import zhconv

from .av_recorder import AVRecorder
from .stt_engine import create_stt_engine
from .wake_word_engine import WakeWordEngine
from .backend import BackendError, create_backend
from .tts import TTSEngine

logger = logging.getLogger("hermes-voice")

# ANSI color codes
_COLORS = {
    logging.DEBUG: "\033[2m",       # dim
    logging.INFO: "\033[0m",        # default
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI colors, filename, and line number."""

    def format(self, record):
        color = _COLORS.get(record.levelno, "")
        # Override levelname to show filename:lineno
        record.colored_levelname = (
            f"{color}[{record.filename}:{record.lineno}]{_RESET}"
        )
        # Build format manually for clean output
        ts = self.formatTime(record, "%H:%M:%S")
        return (
            f"{color}{ts}{_RESET} "
            f"{record.colored_levelname} "
            f"{record.getMessage()}"
        )


def _project_root():
    """项目根目录（voice/ 的上一级），保证任意 CWD 可运行。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    if path is None:
        path = os.path.join(_project_root(), "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("HERMES_API_KEY") or cfg.get("hermes_api_key", "")
    cfg["hermes_api_key"] = api_key
    return cfg


def _to_simplified(text):
    """Convert traditional Chinese to simplified."""
    return zhconv.convert(text, "zh-cn")


def _strip_wake_word(text, keywords):
    """Strip wake word prefix from transcribed text."""
    for kw in sorted(keywords, key=len, reverse=True):
        if text.startswith(kw):
            return text[len(kw):].lstrip("，, ").strip()
    return text


def _is_valid_speech(text):
    """Check if transcribed text has meaningful content (not just cough/noise).

    A cough or background pop may trigger VAD and produce blank or
    punctuation-only transcription (e.g. ``。``). This filters those out
    while allowing single-character responses like ``是`` / ``好``.
    """
    return bool(text and text.strip() and any(c.isalnum() for c in text.strip()))


def play_beep():
    try:
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Blow.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def check_mic_permission():
    """Check macOS microphone permission. Exit with message if denied."""
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeAudio
    )
    if status == AVFoundation.AVAuthorizationStatusDenied:
        logger.critical(
            "麦克风权限被拒绝。请在 系统设置 → 隐私与安全性 → 麦克风 中允许本应用"
        )
        sys.exit(1)
    elif status == AVFoundation.AVAuthorizationStatusRestricted:
        logger.critical(
            "麦克风权限受系统限制 (家长控制/MDM)，无法访问麦克风"
        )
        sys.exit(1)
    elif status == AVFoundation.AVAuthorizationStatusNotDetermined:
        logger.info("请求麦克风权限 …")
        # AVFoundation will prompt on first access, no explicit call needed.
        # The later recorder.start() triggers the system permission dialog.


def validate_config(cfg):
    """Validate required config keys and their types."""
    required = {
        "samplerate": (int, float),
        "silence_timeout": (int, float),
    }
    for key, expected_type in required.items():
        if key not in cfg:
            raise ValueError(f"配置缺少必要字段: {key}")
        if not isinstance(cfg[key], expected_type):
            expected_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else " or ".join(t.__name__ for t in expected_type)
            )
            raise TypeError(
                f"配置字段 {key} 类型错误: "
                f"期望 {expected_name}, 实际 {type(cfg[key]).__name__}"
            )


def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "hermes-voice.log")

    # File handler: plain format with filename:lineno
    file_fmt = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_242_880, backupCount=3
    )
    file_handler.setFormatter(file_fmt)

    # Console handler: colored + filename:lineno
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def main():
    setup_logging()
    config = load_config()

    check_mic_permission()
    validate_config(config)

    logger.info("初始化组件 …")

    recorder = AVRecorder(
        samplerate=config.get("samplerate", 16000),
        silence_timeout=config.get("silence_timeout", 1.5),
        max_record_sec=config.get("max_record_sec", 15),
    )
    stt = create_stt_engine(config)
    wake_words = config.get("kws_keywords", ["小九"])
    kws = WakeWordEngine(
        keywords=wake_words,
        threshold=config.get("kws_threshold", 0.25),
    )
    backend = create_backend(config)
    try:
        backend.prepare()
    except BackendError as exc:
        # 配置级致命错误（如 file 路径不可写 / openai 缺 base_url）：中止
        logger.critical("后端准备失败: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("后端 prepare 异常，继续以当前配置启动")
    logger.info("后端 → %s (%s)", backend.name, getattr(backend, "base_url", ""))
    tts = TTSEngine()

    def _speak_and_recover(recorder, tts, text):
        """Speak TTS reply with barge-in support.

        Returns audio bytes (numpy float32) if user interrupted TTS,
        None if TTS completed normally or text was empty.
        """
        if not text or not text.strip():
            return None

        recorder.set_tts_active(True, config.get("bargein_threshold", 0.55))

        def check_barge_in():
            return recorder.check_interrupt(
                min_duration=config.get("bargein_duration", 0.3)
            )

        try:
            completed = tts.speak(text, interrupt_check=check_barge_in)
        finally:
            recorder.set_tts_active(False)

        if not completed:
            logger.info("TTS 被用户打断")
            # 等待 VAD 完成当前语句（打断触发时用户确实正在说话）
            # 3 秒不够（用户说 2 秒 + VAD 静音确认 1.5 秒 = 3.5 秒）
            # 用 max_record_sec 作为上限，避免背景噪音持续触发 VAD 导致死等
            audio = recorder.wait_utterance(timeout=config.get("max_record_sec", 15))
            return audio

        return None

    def _handle_turn(recorder, stt, backend, text):
        """Route one transcribed utterance through the active backend.

        Returns interrupted audio (speech captured during TTS barge-in) if a
        spoken reply was given and cut short, else None.
        """
        try:
            reply = backend.handle(text)
        except ConnectionError as exc:
            # 后端失败在此吞掉并返回 None，不向上抛：调用方保持当前状态——
            # AWAKE 跟随时窗继续敞开，下一句直接重试、无需重新唤醒；
            # 安静空闲仍会超时自然回到 LISTENING。
            logger.warning("后端不可达 (%s)", exc)
            tts.speak("后端服务不可达，请检查配置或网络")
            return None
        except BackendError as exc:
            logger.error("后端处理失败: %s", exc)
            tts.speak(str(exc) or "处理失败，请查看日志")
            return None

        reply = _renew_if_pending(recorder, stt, backend, reply)

        if reply:
            logger.info("朗读 → %s", reply)
            return _speak_and_recover(recorder, tts, reply)
        # 无文字回复（file 听写）：响一声确认，不朗读
        play_beep()
        return None

    def _process_interruption(recorder, stt, backend, tts, interrupted_audio):
        """Transcribe and respond to audio captured during TTS barge-in.

        Uses the already-captured *interrupted_audio* directly. Handles
        recursive interrupts — if the reply TTS is also barged in, the
        chain continues naturally.
        """
        if interrupted_audio is None:
            return

        text = _to_simplified(stt.transcribe(interrupted_audio))
        if not _is_valid_speech(text):
            return

        logger.info("打断→ %s", text)
        more_audio = _handle_turn(recorder, stt, backend, text)
        if more_audio is not None:
            _process_interruption(recorder, stt, backend, tts, more_audio)

    def _renew_if_pending(recorder, stt, backend, current_reply):
        """If VAD captured speech during last blocking call, re-request.

        Consumes audio VAD stored while ``backend.handle()`` blocked (slow
        LLM / tool calls), transcribes it, and makes a fresh request so the
        assistant answers what the user *actually* just said.
        """
        audio = recorder.consume_utterance()
        if audio is None:
            return current_reply
        text = _to_simplified(stt.transcribe(audio))
        if not text or not text.strip():
            return current_reply
        logger.info("等待期间→ %s", text)
        try:
            return backend.handle(text)
        except ConnectionError:
            return current_reply

    session_timeout = config.get("session_timeout", 30)

    shutdown_requested = False

    def shutdown(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        logger.info("正在关闭 …")
        tts.stop()
        recorder.stop()
        kws.close()
        backend.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    state = "LISTENING"
    kw_display = " / ".join(wake_words)
    logger.info("状态机 → 就绪，说\"%s\"唤醒我", kw_display)

    while not shutdown_requested:
        try:
            if state == "LISTENING":
                recorder.set_wake_hook(kws.process_chunk)

                if not recorder.wait_for_wake_word():
                    continue

                detected = kws.last_keyword
                recorder.set_wake_hook(None)
                kws.reset()
                play_beep()
                greeting = config.get("wake_greeting", "我在")
                if greeting:
                    tts.speak(greeting)

                logger.info("唤醒词 → 检测到\"%s\"，等待指令 …", detected)

                audio = recorder.read_utterance(
                    idle_timeout=session_timeout
                )
                if audio is None:
                    logger.info("状态机 → 没听到指令，回到待唤醒")
                    continue

                text = _to_simplified(stt.transcribe(audio))
                if not _is_valid_speech(text):
                    continue

                logger.info("麦克风→ %s", text)

                interrupted = _handle_turn(recorder, stt, backend, text)
                state = "AWAKE"
                recorder.start()
                logger.info("状态机 → 进入跟随时窗 (%.0f秒)",
                            session_timeout)
                _process_interruption(recorder, stt, backend, tts, interrupted)

            elif state == "AWAKE":
                audio = recorder.read_utterance(
                    idle_timeout=session_timeout
                )

                if audio is None:
                    logger.info("状态机 → 跟随时窗超时，回到待唤醒")
                    backend.clear_context()
                    state = "LISTENING"
                    continue

                text = _to_simplified(stt.transcribe(audio))
                if not _is_valid_speech(text):
                    continue

                logger.info("麦克风→ %s", text)

                interrupted = _handle_turn(recorder, stt, backend, text)
                _process_interruption(recorder, stt, backend, tts, interrupted)

        except KeyboardInterrupt:
            break

        except Exception:
            logger.exception("意外错误，恢复中 …")
            state = "LISTENING"

    recorder.stop()
    backend.close()


if __name__ == "__main__":
    main()
