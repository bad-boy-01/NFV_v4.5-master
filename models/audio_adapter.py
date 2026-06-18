"""
Audio Adapter — Novel Video Factory v4
PRIMARY:  edge-tts (Microsoft Edge TTS, FREE, no API key, very natural voices)
FALLBACK: Silent WAV (keeps pipeline running if TTS fails)

Install: pip install edge-tts
"""
import asyncio
import logging
import os
import struct
import wave

logger = logging.getLogger(__name__)


class LocalAudioAdapter:
    def __init__(self, config: dict = None):
        cfg = config or {}
        audio_cfg = cfg.get("models", {}).get("audio", {})
        self.provider = audio_cfg.get("provider", "edge_tts")
        self.voice = audio_cfg.get("voice", "en-US-AndrewNeural")
        self._edge_tts_warned = False   # log missing edge-tts only once
        logger.info(f"Audio adapter: provider={self.provider}, voice={self.voice}")

    # ── Public Interface ──────────────────────────────────────────────────────
    def generate_audio(self, text: str, output_path: str):
        """Generate speech from text and save as WAV."""
        if not text or not text.strip():
            text = "..."

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if self.provider == "edge_tts":
            if self._edge_tts(text, output_path):
                return

        # Last resort: silence (duration based on word count)
        duration = max(1.5, len(text.split()) * 0.38)
        self._mock_wav(output_path, duration=duration)

    # ── edge-tts ──────────────────────────────────────────────────────────────
    def _edge_tts(self, text: str, output_path: str) -> bool:
        """
        Microsoft Edge TTS — completely free, no API key, very natural voices.
        Generates MP3 then converts to WAV with ffmpeg.
        """
        try:
            import edge_tts  # type: ignore

            tmp_mp3 = output_path.replace(".wav", "_tmp.mp3")

            async def _gen():
                comm = edge_tts.Communicate(text, self.voice)
                await comm.save(tmp_mp3)

            # Run async in a fresh event loop (safe for Kaggle)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_gen())
            finally:
                loop.close()

            # Convert MP3 → WAV with ffmpeg (available on Kaggle/Colab)
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "22050", "-ac", "1", output_path],
                capture_output=True,
            )
            if os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)

            if result.returncode == 0 and os.path.exists(output_path):
                logger.debug(f"edge-tts ✓: {os.path.basename(output_path)}")
                return True
            else:
                logger.warning(f"ffmpeg WAV conversion failed: {result.stderr.decode()[:200]}")
                return False

        except ImportError:
            if not self._edge_tts_warned:
                logger.warning("edge-tts not installed. Run: pip install edge-tts")
                self._edge_tts_warned = True
            return False
        except Exception as e:
            logger.warning(f"edge-tts failed: {e}")
            return False

    # ── Silent WAV fallback ───────────────────────────────────────────────────
    def _mock_wav(self, output_path: str, duration: float = 2.0):
        """Create a silent WAV file so the video pipeline doesn't crash."""
        sample_rate = 22050
        n_frames = int(sample_rate * duration)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
        logger.debug(f"Silent WAV ({duration:.1f}s): {os.path.basename(output_path)}")
