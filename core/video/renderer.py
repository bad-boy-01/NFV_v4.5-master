"""
Video Renderer — Novel Video Factory v4

Assembles images + audio into 10-minute clips, then stitches into final video.

BUG FIXES vs v3:
- torch imported inside functions (not at top-level) to avoid crash on CPU-only
- FFmpeg concat list uses ABSOLUTE paths to fix 'file not found' errors
- Ken Burns effect wrapped in try/except so a bad image doesn't kill the clip
- subtitle rendering uses DejaVu font (always available on Kaggle/Linux)
- clip.close() called in finally block to prevent resource leaks
"""
import gc
import json
import logging
import os
import random
import subprocess
import hashlib
from typing import List

logger = logging.getLogger(__name__)


class VideoRenderer:
    """
    Assembles per-scene images and audio into MP4 clips.
    Each clip is ~10 minutes. Then stitches all clips into a master video.
    """
    def __init__(self, project_dir: str, config: dict = None):
        self.project_dir = project_dir
        cfg = config or {}
        vid_cfg = cfg.get("video", {})
        self.fps = vid_cfg.get("fps", 24)
        self.font = vid_cfg.get("font", "DejaVu-Sans-Bold")
        self.font_size = vid_cfg.get("font_size", 40)

        self.output_dir = os.path.join(project_dir, "output")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.clips_path = os.path.join(self.output_dir, "clips.json")
        self.final_video_path = os.path.join(self.videos_dir, "final_video.mp4")

        os.makedirs(self.videos_dir, exist_ok=True)

    def render(self):
        """Main entry point: renders all clips and stitches final video."""
        if not os.path.exists(self.clips_path):
            logger.error("clips.json not found — cannot render video")
            return

        with open(self.clips_path, "r", encoding="utf-8") as f:
            clips_data = json.load(f)

        logger.info(f"Rendering {len(clips_data)} clips…")
        rendered_clip_paths = []

        for clip in clips_data:
            clip_id = clip["clip_id"]
            clip_path = os.path.join(self.videos_dir, f"{clip_id}.mp4")
            
            # Use a hash of the clip's shots to detect changes
            clip_content_hash = hashlib.sha256(json.dumps(clip.get("shots", []), sort_keys=True).encode()).hexdigest()[:16]
            hash_file = clip_path + ".hash"
            
            existing_hash = ""
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    existing_hash = f.read().strip()

            if os.path.exists(clip_path) and existing_hash == clip_content_hash:
                logger.info(f"Clip exists and unchanged, skipping: {clip_id}")
                rendered_clip_paths.append(clip_path)
                continue

            success = self._render_clip(clip, clip_path)
            if success:
                rendered_clip_paths.append(clip_path)
                with open(hash_file, "w") as f:
                    f.write(clip_content_hash)

        if not rendered_clip_paths:
            logger.error("No clips were rendered successfully")
            return

        self._stitch_final(rendered_clip_paths)
        self.generate_srt(clips_data)

    def generate_srt(self, clips_data: List[dict]):
        """Generate a master SRT file for the entire video."""
        srt_path = os.path.join(self.videos_dir, "subtitles.srt")
        logger.info(f"Generating SRT: {srt_path}")
        
        def format_time(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds % 1) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

        current_time = 0.0
        srt_index = 1
        
        try:
            from moviepy.editor import AudioFileClip
            with open(srt_path, "w", encoding="utf-8") as f:
                for clip in clips_data:
                    for shot in clip.get("shots", []):
                        sid = shot["scene_id"]
                        aud_path = os.path.join(self.audio_dir, f"{sid}.wav")
                        text = shot.get("narration_text", "").strip()
                        
                        if not os.path.exists(aud_path) or not text:
                            continue
                            
                        duration = AudioFileClip(aud_path).duration
                        start_str = format_time(current_time)
                        end_str = format_time(current_time + duration)
                        
                        f.write(f"{srt_index}\n{start_str} --> {end_str}\n{text}\n\n")
                        
                        current_time += duration
                        srt_index += 1
            logger.info("✓ SRT generated")
        except Exception as e:
            logger.error(f"Failed to generate SRT: {e}")

    def _render_clip(self, clip: dict, output_path: str) -> bool:
        """Render one 10-minute clip from its shots."""
        try:
            # Import moviepy 2.x style
            from moviepy import (ImageClip, AudioFileClip,
                                 concatenate_videoclips,
                                 TextClip, CompositeVideoClip, ColorClip)
        except ImportError:
            logger.error("moviepy not installed — cannot render video. "
                         "Run: pip install moviepy>=2.1.1")
            return False

        clip_id = clip["clip_id"]
        shots = clip.get("shots", [])
        logger.info(f"--- Rendering: {clip_id} ({len(shots)} shots) ---")

        shot_clips = []
        for shot in shots:
            sc = self._render_shot(shot, clip_id)
            if sc is not None:
                shot_clips.append(sc)

        if not shot_clips:
            logger.warning(f"No valid shots in {clip_id}")
            return False

        try:
            final = concatenate_videoclips(shot_clips, method="compose")
            final.write_videofile(
                output_path,
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            logger.info(f"✓ Clip saved: {os.path.basename(output_path)}")
            return True
        except Exception as e:
            logger.error(f"Error writing {clip_id}: {e}")
            return False
        finally:
            for sc in shot_clips:
                try:
                    sc.close()
                except Exception:
                    pass
            gc.collect()

    def _render_shot(self, shot: dict, clip_id: str):
        """Render one shot (image + audio + subtitles)."""
        try:
            from moviepy import (ImageClip, AudioFileClip,
                                 TextClip, CompositeVideoClip, ColorClip)
        except ImportError:
            return None

        shot_id = shot["scene_id"]
        img_path = os.path.join(self.images_dir, f"{shot_id}.png")
        aud_path = os.path.join(self.audio_dir, f"{shot_id}.wav")

        if not os.path.exists(img_path):
            logger.warning(f"Missing image: {shot_id} — skipping")
            return None
        if not os.path.exists(aud_path):
            logger.warning(f"Missing audio: {shot_id} — skipping")
            return None

        try:
            audio_clip = AudioFileClip(aud_path)
            duration = max(audio_clip.duration, 1.0)

            # Ken Burns zoom effect
            zoom_dir = random.choice(["in", "out"])
            zoom_speed = 0.03

            def zoom_effect(t):
                if zoom_dir == "in":
                    return 1 + zoom_speed * t / duration
                else:
                    return 1 + zoom_speed - zoom_speed * t / duration

            img_clip = (ImageClip(img_path)
                        .with_duration(duration)
                        .resized(zoom_effect)
                        .with_position("center"))

            # Subtitles
            subtitle_text = shot.get("narration_text", "").strip()
            if subtitle_text:
                try:
                    img_clip = self._add_subtitles(img_clip, subtitle_text, duration)
                except Exception as te:
                    logger.debug(f"Subtitle failed for {shot_id}: {te}")

            return img_clip.with_audio(audio_clip)

        except Exception as e:
            logger.error(f"Failed to render shot {shot_id}: {e}")
            return None

    def _add_subtitles(self, img_clip, text: str, duration: float):
        """
        Add subtitle overlay to an ImageClip using MoviePy 2.x API.
        """
        from moviepy import TextClip, CompositeVideoClip, ColorClip

        words = text.split()
        if not words:
            return img_clip

        w = img_clip.w
        box_w = int(w * 0.85)
        # Rough estimate of average glyph advance for a bold sans font:
        # ~0.55x the font size in pixels.
        chars_per_line = max(10, int(box_w / (self.font_size * 0.55)))
        max_chars = chars_per_line * 2  # target: ~2 lines per caption group

        groups: List[str] = []
        group_words: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            added_len = len(word) + (1 if current else 0)
            if current and current_len + added_len > max_chars:
                groups.append(" ".join(current))
                group_words.append(current)
                current, current_len = [], 0
                added_len = len(word)
            current.append(word)
            current_len += added_len
        if current:
            groups.append(" ".join(current))
            group_words.append(current)

        if not groups:
            return img_clip

        total_words = sum(len(gw) for gw in group_words) or 1
        subtitle_clips = []
        t_cursor = 0.0

        for idx, (g_text, gw) in enumerate(zip(groups, group_words)):
            is_last = idx == len(groups) - 1
            if is_last:
                g_dur = max(0.1, duration - t_cursor)  # absorb rounding drift
            else:
                g_dur = max(0.1, duration * (len(gw) / total_words))

            txt = TextClip(
                text=g_text,
                font=self.font,
                font_size=self.font_size,
                color="white",
                method="caption",
                size=(box_w, None),
                text_align="center",
            ).with_duration(g_dur).with_start(t_cursor)

            bg_h = txt.h + 30
            bg = (ColorClip(size=(w, bg_h), color=(0, 0, 0))
                  .with_opacity(0.45)
                  .with_duration(g_dur)
                  .with_start(t_cursor)
                  .with_position(("center", "bottom")))

            txt = txt.with_position(("center", img_clip.h - bg_h + 15))
            subtitle_clips.extend([bg, txt])
            t_cursor += g_dur

        return CompositeVideoClip([img_clip] + subtitle_clips)


    def _stitch_final(self, clip_paths: List[str]):
        """
        Use FFmpeg to stitch all rendered clips into the final master video.
        BUG FIX: Uses absolute paths in the concat list to prevent 'file not found' errors.
        """
        if len(clip_paths) == 1:
            import shutil
            shutil.copy2(clip_paths[0], self.final_video_path)
            logger.info(f"Single clip → final: {self.final_video_path}")
            return

        list_path = os.path.join(self.videos_dir, "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                # BUG FIX: Use absolute path to avoid FFmpeg cwd issues
                abs_path = os.path.abspath(cp).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_path, "-c", "copy", self.final_video_path],
                check=True,
                capture_output=True,
            )
            os.remove(list_path)
            logger.info(f"✓ Final video: {self.final_video_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg stitch failed: {e.stderr.decode()[:500]}")
        except Exception as e:
            logger.error(f"Stitch error: {e}")
