"""
Image Adapter — Novel Video Factory v4
Model: cagliostrolab/animagine-xl-3.1 (FREE on HuggingFace, no API key)

This is the BEST free model for Korean manhwa / webtoon style.
It uses Animagine-XL 3.1 score tags (score_9, score_8_up) for quality.

Features:
- DPM++ 2M Karras scheduler (30% faster than DDIM at same quality)
- IP-Adapter for character reference images (consistency across scenes)
- Multi-pose character sheet generation (6 poses per character)
- Quality filter with auto-retry on bad outputs
- VRAM-safe: CPU offload, VAE slicing/tiling
- 832×480 default (16:9, fast on T4) or 1344×768 (higher quality)
"""
import gc
import hashlib
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# 6 emotional poses for character reference sheets
POSE_PROMPTS = {
    "front":  "full body, standing, neutral expression, arms at sides, front view, white background",
    "smile":  "upper body, smiling happily, slight head tilt, warm expression",
    "angry":  "upper body, angry expression, furrowed brows, clenched fists, intense glare",
    "crying": "upper body, crying, tears streaming, sad expression, downcast eyes",
    "fight":  "full body, combat stance, dynamic pose, raised fist, battle ready, fierce expression",
    "sit":    "full body, sitting cross-legged, calm expression, eyes closed, meditative",
}

# Animagine-XL quality prefix — these MUST come first for best results
ANIMAGINE_QUALITY = "score_9, score_8_up, (masterpiece:1.2), (best quality:1.1), highres"

# Master negative prompt for Animagine-XL
MASTER_NEGATIVE = (
    "score_6, score_5, score_4, "
    "(worst quality, low quality:1.4), bad anatomy, bad hands, "
    "text, error, missing fingers, extra digit, fewer digits, "
    "cropped, jpeg artifacts, signature, watermark, username, "
    "blurry, ugly, deformed, 3d render, photo, photorealistic, "
    "western comic, american comic, fat, extra limbs, cloned face, "
    "mutation, fused fingers, long neck"
)


class LocalImageAdapter:
    """
    Animagine-XL 3.1 image generator — 100% free, no API key.
    Optimised for Kaggle T4 GPU (16GB VRAM).
    """
    def __init__(self, config: dict = None):
        cfg = config or {}
        img_cfg = cfg.get("models", {}).get("image", {})

        self.model_name = img_cfg.get("model", "cagliostrolab/animagine-xl-3.1")
        self.width = img_cfg.get("width", 832)
        self.height = img_cfg.get("height", 480)
        self.steps = img_cfg.get("num_inference_steps", 20)
        self.guidance_scale = img_cfg.get("guidance_scale", 7.0)
        self.use_fast_scheduler = img_cfg.get("use_fast_scheduler", True)
        self.use_ip_adapter = img_cfg.get("use_ip_adapter", True)
        # Was hardcoded to 0.35 — too weak for the generic (non-face-specialized)
        # IP-Adapter weight to reliably lock identity across independent
        # generations. Raised default; still overridable via config.
        self.ip_adapter_scale = img_cfg.get("ip_adapter_scale", 0.65)
        # ip-adapter-plus-face is specifically tuned for facial identity
        # preservation (the generic ip-adapter_sdxl.bin is tuned for overall
        # style/composition transfer, which is weaker for "is this the same
        # character" consistency). Both ship from the same h94/IP-Adapter repo.
        self.ip_adapter_weight_name = img_cfg.get(
            "ip_adapter_weight_name", "ip-adapter-plus-face_sdxl_vit-h.bin"
        )

        qf_cfg = cfg.get("quality_filter", {})
        self.max_retries = qf_cfg.get("max_retries", 3)
        self.min_file_size_kb = qf_cfg.get("min_file_size_kb", 10)
        self.quality_filter_enabled = qf_cfg.get("enabled", True)

        self.pipeline = None
        self._ip_adapter_loaded = False
        self._compel = None  # None = not yet attempted, False = tried and unavailable
        self._init_pipeline()

    # ── Pipeline Init ─────────────────────────────────────────────────────────
    def _init_pipeline(self):
        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            logger.info(f"Loading image model: {self.model_name}")
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            # DPM++ 2M Karras — ~30% faster than DDIM at same quality
            if self.use_fast_scheduler:
                try:
                    from diffusers import DPMSolverMultistepScheduler
                    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipeline.scheduler.config,
                        use_karras_sigmas=True,
                        algorithm_type="dpmsolver++",
                    )
                    logger.info("Scheduler: DPM++ 2M Karras ✓")
                except Exception as e:
                    logger.warning(f"Scheduler swap failed: {e}")

            # Memory optimisations for 16GB T4 VRAM
            try:
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_vae_tiling()
            except Exception:
                pass

            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers ✓")
            except Exception:
                pass

            # Critical: CPU offload keeps 16GB VRAM safe
            self.pipeline.enable_model_cpu_offload()
            logger.info(f"Pipeline ready: {self.model_name} | "
                        f"{self.width}×{self.height} | {self.steps} steps")

        except ImportError as e:
            logger.warning(f"diffusers/torch not installed ({e}) — MOCK mode active")
            self.pipeline = None
        except Exception as e:
            logger.error(f"Pipeline init failed: {e} — MOCK mode active")
            self.pipeline = None

    def _ensure_ip_adapter(self):
        """Load IP-Adapter on first use (lazy-loading saves VRAM)."""
        if self._ip_adapter_loaded or self.pipeline is None:
            return
        if not self.use_ip_adapter:
            return
        # Try the configured weight first (default: face-specialized variant,
        # better at "is this the same character" than the generic weight that
        # was previously hardcoded — generic IP-Adapter is tuned more for
        # overall style/composition transfer than facial identity lock).
        # Fall back to the generic weight if it's unavailable for any reason,
        # rather than disabling IP-Adapter entirely.
        candidates = [self.ip_adapter_weight_name]
        if self.ip_adapter_weight_name != "ip-adapter_sdxl.bin":
            candidates.append("ip-adapter_sdxl.bin")
        for weight_name in candidates:
            try:
                self.pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name=weight_name,
                )
                self.pipeline.set_ip_adapter_scale(self.ip_adapter_scale)
                # Re-enable CPU offload so the new IP-Adapter encoder is handled correctly
                self.pipeline.enable_model_cpu_offload()
                self._ip_adapter_loaded = True
                logger.info(
                    f"IP-Adapter loaded ✓ ({weight_name}, scale={self.ip_adapter_scale}) "
                    f"— character consistency enabled"
                )
                return
            except Exception as e:
                logger.warning(f"IP-Adapter weight '{weight_name}' failed to load: {e}")
        logger.warning("IP-Adapter unavailable on any candidate weight — consistency via prompts only")

    def _ensure_compel(self):
        """
        Lazy-load Compel for long-prompt support. SDXL's two CLIP text
        encoders have a hard 77-token limit each — by default diffusers
        truncates anything past that silently, which is what was dropping
        the "korean manhwa style" tags on nearly every image (see
        FALLBACK_FAILURE_ANALYSIS.md). Compel encodes a prompt in 77-token
        chunks and concatenates the resulting embeddings, so nothing is
        dropped regardless of prompt length.
        """
        if self._compel is not None or self.pipeline is None:
            return
        try:
            from compel import Compel, ReturnedEmbeddingsType
            self._compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
            )
            logger.info("Compel long-prompt encoder loaded ✓ (prompts >77 tokens are chunked, not truncated)")
        except Exception as e:
            logger.warning(
                f"Compel unavailable ({e}) — falling back to plain prompt strings "
                f"(prompter.py's token-budget trimming is the only guard against "
                f"77-token truncation in this mode)"
            )
            self._compel = False  # sentinel: tried once, don't retry every call

    # ── Public Interface ──────────────────────────────────────────────────────
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "",
        reference_image_paths: List[str] = None,
        seed: int = None,
        generation_params: dict = None,
    ):
        """
        Generate one image and save to output_path.
        Retries up to max_retries times if quality filter rejects it.
        """
        params = generation_params or {}
        effective_seed = params.get("seed", seed)
        steps = params.get("steps", self.steps)
        cfg = params.get("cfg", self.guidance_scale)
        w = params.get("width", self.width)
        h = params.get("height", self.height)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # In mock mode (no GPU/diffusers) placeholders are tiny — skip QA entirely
        use_qa = self.quality_filter_enabled and (self.pipeline is not None)

        for attempt in range(self.max_retries):
            s = ((effective_seed + attempt * 7919) % (2**31 - 1)
                 if effective_seed is not None else 42)
            if attempt > 0:
                logger.info(f"  Retry {attempt}/{self.max_retries - 1} (seed={s})")

            self._run_generation(prompt, output_path, negative_prompt,
                                 reference_image_paths, s, steps, cfg, w, h)

            if not use_qa or self._passes_quality(output_path):
                return  # accepted ✓

        if use_qa:
            logger.warning(f"All {self.max_retries} attempts failed QA filter: {output_path}")

    def _run_generation(self, prompt, output_path, negative_prompt,
                        ref_paths, seed, steps, cfg, w, h):
        if self.pipeline is None:
            logger.info(f"[MOCK] {prompt[:60]}…")
            self._save_placeholder(output_path, w, h)
            return

        import torch
        self._ensure_compel()
        kwargs = None

        if self._compel:
            try:
                conditioning, pooled = self._compel(prompt)
                neg_conditioning, neg_pooled = self._compel(negative_prompt or MASTER_NEGATIVE)
                [conditioning, neg_conditioning] = self._compel.pad_conditioning_tensors_to_same_length(
                    [conditioning, neg_conditioning]
                )
                kwargs = {
                    "prompt_embeds": conditioning,
                    "pooled_prompt_embeds": pooled,
                    "negative_prompt_embeds": neg_conditioning,
                    "negative_pooled_prompt_embeds": neg_pooled,
                    "width": w,
                    "height": h,
                    "num_inference_steps": steps,
                    "guidance_scale": cfg,
                }
            except Exception as e:
                logger.warning(f"  Compel encoding failed ({e}) — using plain prompt for this image")
                kwargs = None

        if kwargs is None:
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or MASTER_NEGATIVE,
                "width": w,
                "height": h,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
            }

        # Deterministic seed
        try:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        except Exception:
            pass

        # IP-Adapter: use reference images for character consistency
        if ref_paths and self.use_ip_adapter:
            valid = [p for p in ref_paths if p and os.path.exists(p)]
            if valid:
                self._ensure_ip_adapter()
                if self._ip_adapter_loaded:
                    try:
                        from PIL import Image
                        import numpy as np
                        imgs = [Image.open(p).convert("RGB") for p in valid]
                        if len(imgs) > 1:
                            # Average multiple reference images for multi-character scenes
                            avg = np.mean(
                                [np.array(img.resize((224, 224))) for img in imgs], axis=0
                            ).astype("uint8")
                            kwargs["ip_adapter_image"] = Image.fromarray(avg)
                        else:
                            kwargs["ip_adapter_image"] = imgs[0]
                        logger.debug(f"  IP-Adapter: {len(valid)} ref image(s)")
                    except Exception as e:
                        logger.warning(f"  Reference injection failed: {e}")

        try:
            image = self.pipeline(**kwargs).images[0]
            image.save(output_path)
            logger.info(f"  ✓ Saved: {os.path.basename(output_path)}")
        except Exception as e:
            logger.error(f"  Generation error: {e}")
            self._save_placeholder(output_path, w, h)

    # ── Multi-pose Character Sheet ────────────────────────────────────────────
    def generate_character_sheet(
        self,
        char_id: str,
        char_name: str,
        dna_str: str,
        output_dir: str,
        negative_prompt: str = "",
        poses: list = None,
    ):
        """
        Generates 6 reference poses for a character.
        Creates: output_dir/{char_id}/{pose}.png
        These are used by IP-Adapter for consistency across all scenes.
        """
        if poses is None:
            poses = list(POSE_PROMPTS.keys())

        pose_dir = os.path.join(output_dir, char_id)
        os.makedirs(pose_dir, exist_ok=True)

        seed_base = abs(hash(char_id + char_name)) % (2**31 - 1)

        # Detect gender for proper subject tag
        dna_lower = dna_str.lower()
        gender_tag = ("1girl"
                      if any(w in dna_lower for w in ["girl", "woman", "female", "she"])
                      else "1boy")

        for i, pose in enumerate(poses):
            out_path = os.path.join(pose_dir, f"{pose}.png")
            if os.path.exists(out_path):
                logger.info(f"  Pose exists: {char_name}/{pose} — skipping")
                continue

            pose_tags = POSE_PROMPTS[pose]
            prompt = (
                f"{ANIMAGINE_QUALITY}, "
                f"{gender_tag}, {dna_str}, {pose_tags}, "
                f"character reference sheet, "
                f"manhwa, webtoon, korean manhwa style, sharp lineart"
            )
            neg = (
                f"score_6, score_5, (worst quality, low quality:1.4), "
                f"bad anatomy, blurry, text, watermark, {negative_prompt}"
            )
            pose_seed = (seed_base + i * 1013) % (2**31 - 1)
            logger.info(f"  Generating {char_name}/{pose}…")

            # Square format for character sheets (better face detail)
            self.generate_image(
                prompt, out_path, negative_prompt=neg,
                seed=pose_seed,
                generation_params={"width": 768, "height": 768, "steps": 25, "cfg": 7.0},
            )

    # ── Quality Filter ────────────────────────────────────────────────────────
    def _passes_quality(self, output_path: str) -> bool:
        """Basic quality check: file exists and is large enough to be a real image."""
        if not os.path.exists(output_path):
            return False
        size_kb = os.path.getsize(output_path) / 1024
        if size_kb < self.min_file_size_kb:
            logger.warning(f"  QA fail: file too small ({size_kb:.1f}KB < {self.min_file_size_kb}KB)")
            return False
        return True

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _save_placeholder(self, output_path: str, w: int = 832, h: int = 480):
        """Create a visible placeholder so downstream stages don't crash."""
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (w, h), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)
            draw.text((w // 2 - 80, h // 2 - 10),
                      "[ IMAGE GENERATION FAILED ]", fill=(180, 80, 80))
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            img.save(output_path)
        except Exception as e:
            logger.error(f"Placeholder save failed: {e}")
            with open(output_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header as last resort

    def unload(self):
        """Free GPU + CPU RAM before video rendering."""
        if self.pipeline is not None:
            logger.info("Unloading image pipeline from VRAM…")
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
            del self.pipeline
            self.pipeline = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared ✓")
        except ImportError:
            pass

    def cleanup(self):
        """Alias for compatibility."""
        self.unload()
