"""TTS service using model and voice managers."""

import asyncio
import os
import re
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from ..inference.kokoro_v1 import KokoroV1
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from ..structures.schemas import NormalizationOptions
from .audio import AudioNormalizer, AudioService
from .streaming_audio_writer import StreamingAudioWriter
from .text_processing import tokenize
from .text_processing.text_processor import process_text_chunk, smart_split


class TTSService:
    """Text-to-speech service."""

    # More aggressive GPU utilization
    _chunk_semaphore = asyncio.Semaphore(48)  # Increased from 32 to 48
    _io_semaphore = asyncio.Semaphore(24)     # Increased I/O concurrency
    
    # Enhanced multi-level caching
    _voice_cache = {}
    _path_cache = {}
    _tensor_pool = {}
    _pipeline_cache = {}  # Cache for text processing
    
    # Larger thread pool
    _thread_pool = ThreadPoolExecutor(max_workers=8)  # Increased from 4
    
    # Pipeline reuse cache
    _last_voice_config = None
    _last_normalizer = None

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> "TTSService":
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        # Pre-warm more aggressively
        await service._prewarm_system()
        return service

    async def _prewarm_system(self):
        """Aggressively pre-load everything we can."""
        try:
            # Pre-warm more voices
            common_voices = [
                settings.default_voice, "af_bella", "af_sarah", "af_nicole", "af_heart",
                "am_adam", "am_michael", "bm_daniel", "bm_george", "bf_emma"
            ]
            backend = self.model_manager.get_backend()
            
            # Load voices in parallel
            tasks = []
            for voice_name in common_voices:
                task = asyncio.create_task(self._prewarm_voice(voice_name, backend))
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Pre-compile regex patterns
            self._voice_split_pattern = re.compile(r"([-+])")
            self._voice_weight_pattern = re.compile(r"(.+?)\(([0-9.]+)\)")
            
            # Pre-warm normalizers for common configurations
            for _ in range(3):
                AudioNormalizer()
                
            logger.info(f"Pre-warmed {len(self._voice_cache)} voices and system components")
            
        except Exception as e:
            logger.debug(f"System pre-warming failed: {e}")

    async def _prewarm_voice(self, voice_name: str, backend):
        """Pre-warm a single voice."""
        try:
            cache_key = f"{voice_name}_{backend.device}"
            if cache_key not in self._voice_cache:
                voice_tensor = await self._voice_manager.load_voice(
                    voice_name, device=backend.device
                )
                self._voice_cache[cache_key] = voice_tensor
                
                # Also cache the path
                path = await self._voice_manager.get_voice_path(voice_name)
                if path:
                    self._path_cache[voice_name] = path
                    
                logger.debug(f"Pre-warmed voice: {voice_name}")
        except Exception as e:
            logger.debug(f"Could not pre-warm voice {voice_name}: {e}")

    async def _process_chunk(
        self,
        chunk_text: str,
        tokens: List[int],
        voice_name: str,
        voice_path: str,
        speed: float,
        writer: StreamingAudioWriter,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        normalizer: Optional[AudioNormalizer] = None,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Process tokens into audio with maximum performance."""
        async with self._chunk_semaphore:
            try:
                # Handle stream finalization (PRESERVED ORIGINAL LOGIC)
                if is_last:
                    if not output_format:
                        yield AudioChunk(np.array([], dtype=np.int16), output=b"")
                        return
                    chunk_data = await AudioService.convert_audio(
                        AudioChunk(np.array([], dtype=np.float32)),
                        output_format, writer, speed, "",
                        normalizer=normalizer, is_last_chunk=True,
                    )
                    yield chunk_data
                    return

                # Skip empty chunks early
                if not tokens and not chunk_text:
                    return

                # Get backend (cached)
                backend = self.model_manager.get_backend()

                # Generate audio with optimizations
                if isinstance(backend, KokoroV1):
                    chunk_index = 0
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        # Process output immediately to reduce memory pressure
                        if output_format:
                            try:
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data, output_format, writer, speed, chunk_text,
                                    is_last_chunk=is_last, normalizer=normalizer,
                                )
                                yield chunk_data
                            except Exception as e:
                                logger.error(f"Failed to convert audio: {str(e)}")
                        else:
                            chunk_data = AudioService.trim_audio(
                                chunk_data, chunk_text, speed, is_last, normalizer
                            )
                            yield chunk_data
                        chunk_index += 1
                        
                        # Clear references to help GC
                        del chunk_data
                else:
                    # Ultra-fast voice tensor lookup
                    cache_key = f"{voice_name}_{backend.device}"
                    if cache_key not in self._voice_cache:
                        # Parallel load if not cached
                        async with self._io_semaphore:
                            if cache_key not in self._voice_cache:
                                self._voice_cache[cache_key] = await self._voice_manager.load_voice(
                                    voice_name, device=backend.device
                                )
                                logger.debug(f"Hot-loaded voice tensor for {voice_name}")
                    
                    voice_tensor = self._voice_cache[cache_key]
                    
                    # Generate audio
                    chunk_data = await self.model_manager.generate(
                        tokens, voice_tensor, speed=speed, return_timestamps=return_timestamps,
                    )

                    if chunk_data.audio is None or len(chunk_data.audio) == 0:
                        logger.error("Model generated invalid audio chunk")
                        return

                    # Process output (PRESERVED ORIGINAL LOGIC)
                    if output_format:
                        try:
                            chunk_data = await AudioService.convert_audio(
                                chunk_data, output_format, writer, speed, chunk_text,
                                normalizer=normalizer, is_last_chunk=is_last,
                            )
                            yield chunk_data
                        except Exception as e:
                            logger.error(f"Failed to convert audio: {str(e)}")
                    else:
                        trimmed = AudioService.trim_audio(
                            chunk_data, chunk_text, speed, is_last, normalizer
                        )
                        yield trimmed
            except Exception as e:
                logger.error(f"Failed to process tokens: {str(e)}")

    async def _load_voice_from_path_fast(self, path: str, weight: float):
        """Ultra-fast voice loading with aggressive caching."""
        if not path:
            raise ValueError(f"Voice not found at path: {path}")

        # Multi-level cache check
        cache_key = f"path_{path}_{weight}"
        if cache_key in self._tensor_pool:
            return self._tensor_pool[cache_key]

        # Use thread pool with higher priority for voice loading
        loop = asyncio.get_event_loop()
        tensor = await loop.run_in_executor(
            self._thread_pool, 
            lambda: torch.load(path, map_location="cpu", weights_only=True) * weight
        )
        
        # Aggressive caching with size management
        if len(self._tensor_pool) < 100:  # Increased cache size
            self._tensor_pool[cache_key] = tensor
        elif len(self._tensor_pool) < 150:
            # Keep most recently used
            oldest_key = next(iter(self._tensor_pool))
            del self._tensor_pool[oldest_key]
            self._tensor_pool[cache_key] = tensor
        
        return tensor

    async def _get_voices_path_fast(self, voice: str) -> Tuple[str, str]:
        """Blazingly fast voice path resolution."""
        try:
            # Check cache first
            if voice in self._path_cache:
                return voice, self._path_cache[voice]
            
            # Use pre-compiled regex
            split_voice = self._voice_split_pattern.split(voice)

            # Single voice fast path
            if len(split_voice) == 1:
                if ("(" not in voice and ")" not in voice) or settings.voice_weight_normalization:
                    # Try cache first, then load
                    if voice not in self._path_cache:
                        async with self._io_semaphore:
                            path = await self._voice_manager.get_voice_path(voice)
                            if not path:
                                raise RuntimeError(f"Voice not found: {voice}")
                            self._path_cache[voice] = path
                    
                    path = self._path_cache[voice]
                    logger.debug(f"Using cached voice path: {path}")
                    return voice, path

            # Combined voice processing with enhanced caching
            cache_key = f"combined_{voice}"
            if cache_key in self._voice_cache:
                return voice, self._voice_cache[cache_key]

            # Fast weight parsing
            total_weight = 0
            for voice_index in range(0, len(split_voice), 2):
                voice_object = split_voice[voice_index]
                match = self._voice_weight_pattern.match(voice_object)
                if match:
                    voice_name = match.group(1)
                    voice_weight = float(match.group(2))
                else:
                    voice_name = voice_object
                    voice_weight = 1
                total_weight += voice_weight
                split_voice[voice_index] = (voice_name, voice_weight)

            if not settings.voice_weight_normalization:
                total_weight = 1

            # Parallel voice loading for combined voices
            async def load_voice_parallel(voice_name, weight):
                if voice_name not in self._path_cache:
                    path = await self._voice_manager.get_voice_path(voice_name)
                    self._path_cache[voice_name] = path
                else:
                    path = self._path_cache[voice_name]
                return await self._load_voice_from_path_fast(path, weight / total_weight)

            # Load first voice
            combined_tensor = await load_voice_parallel(split_voice[0][0], split_voice[0][1])

            # Process remaining voices
            for operation_index in range(1, len(split_voice) - 1, 2):
                voice_tensor = await load_voice_parallel(
                    split_voice[operation_index + 1][0], 
                    split_voice[operation_index + 1][1]
                )
                
                if split_voice[operation_index] == "+":
                    combined_tensor += voice_tensor
                else:
                    combined_tensor -= voice_tensor

            # Fast save with thread pool
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, f"voice_{hash(voice)}.pt")  # Use hash for filename
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._thread_pool, 
                lambda: torch.save(combined_tensor, combined_path)
            )
            
            # Cache aggressively
            self._voice_cache[cache_key] = combined_path
            self._path_cache[voice] = combined_path
            
            return voice, combined_path
            
        except Exception as e:
            logger.error(f"Failed to get voice path: {e}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Ultra-fast audio streaming with maximum optimizations."""
        # Reuse normalizer if same config
        config_key = f"{output_format}_{speed}"
        if config_key == self._last_voice_config and self._last_normalizer:
            stream_normalizer = self._last_normalizer
        else:
            stream_normalizer = AudioNormalizer()
            self._last_normalizer = stream_normalizer
            self._last_voice_config = config_key
            
        chunk_index = 0
        current_offset = 0.0
        
        try:
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voices_path_fast(voice)  # Fast path lookup
            
            # Cache voice paths for repeated use
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream")

            # Process text with aggressive chunking
            async for chunk_text, tokens in smart_split(
                text, lang_code=pipeline_lang_code, normalization_options=normalization_options,
            ):
                try:
                    # Process chunk with maximum concurrency
                    async for chunk_data in self._process_chunk(
                        chunk_text, tokens, voice_name, voice_path, speed, writer,
                        output_format, is_first=(chunk_index == 0), is_last=False,
                        normalizer=stream_normalizer, lang_code=pipeline_lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        if chunk_data.word_timestamps is not None:
                            for timestamp in chunk_data.word_timestamps:
                                timestamp.start_time += current_offset
                                timestamp.end_time += current_offset

                        current_offset += len(chunk_data.audio) / 24000

                        if chunk_data.output is not None:
                            yield chunk_data
                        else:
                            logger.warning(f"No audio generated for chunk: '{chunk_text[:100]}...'")
                        chunk_index += 1
                        
                        # Quick GC hint for large batches
                        if chunk_index % 10 == 0:
                            gc.collect() if torch.cuda.is_available() else None
                            
                except Exception as e:
                    logger.error(f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}")
                    continue

            # Finalize stream
            if chunk_index > 0:
                try:
                    async for chunk_data in self._process_chunk(
                        "", [], voice_name, voice_path, speed, writer, output_format,
                        is_first=False, is_last=True, normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,
                    ):
                        if chunk_data.output is not None:
                            yield chunk_data
                except Exception as e:
                    logger.error(f"Failed to finalize audio stream: {str(e)}")

        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise e

    async def generate_audio(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        return_timestamps: bool = False,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        lang_code: Optional[str] = None,
    ) -> AudioChunk:
        """Generate complete audio for text using streaming internally."""
        audio_data_chunks = []

        try:
            async for audio_stream_data in self.generate_audio_stream(
                text, voice, writer, speed=speed,
                normalization_options=normalization_options,
                return_timestamps=return_timestamps,
                lang_code=lang_code, output_format=None,
            ):
                if len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)

            combined_audio_data = AudioChunk.combine(audio_data_chunks)
            return combined_audio_data
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> torch.Tensor:
        """Combine multiple voices."""
        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()

    async def generate_from_phonemes(
        self,
        phonemes: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate audio directly from phonemes."""
        start_time = time.time()
        try:
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voices_path_fast(voice)  # Use fast path

            if isinstance(backend, KokoroV1):
                result = None
                pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
                logger.info(f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme pipeline")

                try:
                    for r in backend._get_pipeline(pipeline_lang_code).generate_from_tokens(
                        tokens=phonemes, voice=voice_path, speed=speed,
                    ):
                        if r.audio is not None:
                            result = r
                            break
                except Exception as e:
                    logger.error(f"Failed to generate from phonemes: {e}")
                    raise RuntimeError(f"Phoneme generation failed: {e}")

                if result is None or result.audio is None:
                    raise ValueError("No audio generated")

                processing_time = time.time() - start_time
                return result.audio.numpy(), processing_time
            else:
                raise ValueError("Phoneme generation only supported with Kokoro V1 backend")

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise