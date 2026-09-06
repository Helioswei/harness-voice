import logging
import objc
from Foundation import NSObject, NSRunLoop, NSDate
import AVFoundation

logger = logging.getLogger("harness-voice")


class SpeechDelegate(NSObject):
    """Delegate that signals when AVSpeechSynthesizer finishes speaking."""

    def init(self):
        self = objc.super(SpeechDelegate, self).init()
        self.finished = False
        self._current_utterance = None
        return self

    def speechSynthesizer_didFinishSpeechUtterance_(self, synthesizer, utterance):
        if self._current_utterance is None or utterance is self._current_utterance:
            self.finished = True


class TTSEngine:
    """macOS AVSpeechSynthesizer wrapper.

    Uses pyobjc to call the native macOS speech synthesizer.
    ``speak()`` blocks until the utterance finishes.
    """

    _shared_delegate = None

    def __init__(self):
        self.synthesizer = AVFoundation.AVSpeechSynthesizer.alloc().init()
        if TTSEngine._shared_delegate is None:
            TTSEngine._shared_delegate = SpeechDelegate.alloc().init()
        self.synthesizer.setDelegate_(TTSEngine._shared_delegate)

    def speak(self, text, interrupt_check=None):
        """Speak *text* and block until done.

        Parameters
        ----------
        text : str
            Text to speak aloud.
        interrupt_check : callable or None
            Optional zero-arg callback called each iteration. If it returns
            True, speech is stopped immediately and the method returns False.

        Returns
        -------
        bool
            True if utterance completed naturally, False if interrupted.
        """
        if not text or not text.strip():
            return True

        utterance = AVFoundation.AVSpeechUtterance.speechUtteranceWithString_(
            text
        )
        voice = AVFoundation.AVSpeechSynthesisVoice.voiceWithLanguage_("zh-CN")
        utterance.setVoice_(voice)

        delegate = self.synthesizer.delegate()
        delegate._current_utterance = utterance
        delegate.finished = False

        self.synthesizer.speakUtterance_(utterance)

        while not delegate.finished:
            if interrupt_check:
                try:
                    if interrupt_check():
                        self.stop()
                        logger.debug("TTS 被用户打断")
                        return False
                except Exception:
                    # 回调异常时不打断，保持 TTS 继续播放
                    pass
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )
        return True

    def stop(self):
        self.synthesizer.stopSpeakingAtBoundary_(
            AVFoundation.AVSpeechBoundaryImmediate
        )
