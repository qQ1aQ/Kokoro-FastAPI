"""TTS service using model and voice managers."""

import asyncio
import os
import re
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

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

    # Aggressive concurrent processing for GPU utilization
    _chunk_semaphore = asyncio.Semaphore(64)  # High concurrency   #32
    _io_semaphore = asyncio.Semaphore(32)     # Separate semaphore for I/O operations  #16
    
    # Enhanced caching
    _voice_cache = {}
    _path_cache = {}  # Cache for voice paths
    _tensor_pool = {}  # Pre-loaded tensor pool
    
    # Thread pool for CPU-bound operations  #4
    _thread_pool = ThreadPoolExecutor(max_workers=8)

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
        # Pre-warm common voices
        await service._prewarm_default_voices()
        return service

    async def _prewarm_default_voices(self):
        """Pre-load commonly used voices into cache."""
        try:
            common_voices = [
                settings.default_voice,
                "af_bella", "af_sarah", "am_adam", "bm_daniel"
            ]
            backend = self.model_manager.get_backend()
            
            for voice_name in common_voices:
                try:
                    cache_key = f"{voice_name}_{backend.device}"
                    if cache_key not in self._voice_cache:
                        voice_tensor = await self._voice_manager.load_voice(
                            voice_name, device=backend.device
                        )
                        self._voice_cache[cache_key] = voice_tensor
                        logger.debug(f"Pre-warmed voice: {voice_name}")
                except Exception as e:
                    logger.debug(f"Could not pre-warm voice {voice_name}: {e}")
        except Exception as e:
            logger.debug(f"Voice pre-warming failed: {e}")

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
        """Process tokens into audio with enhanced performance."""
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

                # Skip empty chunks
                if not tokens and not chunk_text:
                    return

                # Get backend
                backend = self.model_manager.get_backend()

                # Generate audio using optimized caching
                if isinstance(backend, KokoroV1):
                    chunk_index = 0
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        # For streaming, convert to bytes (PRESERVED ORIGINAL LOGIC)
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
                else:
                    # Enhanced voice tensor caching with device-specific pools
                    cache_key = f"{voice_name}_{backend.device}"
                    if cache_key not in self._voice_cache:
                        # Load voice tensor asynchronously
                        async with self._io_semaphore:
                            if cache_key not in self._voice_cache:  # Double-check
                                self._voice_cache[cache_key] = await self._voice_manager.load_voice(
                                    voice_name, device=backend.device
                                )
                                logger.debug(f"Cached voice tensor for {voice_name} on {backend.device}")
                    
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

    async def _load_voice_from_path(self, path: str, weight: float):
        """Load voice tensor with caching."""
        if not path:
            raise ValueError(f"Voice not found at path: {path}")

        # Check cache first
        cache_key = f"path_{path}_{weight}"
        if cache_key in self._tensor_pool:
            return self._tensor_pool[cache_key]

        logger.debug(f"Loading voice tensor from path: {path}")
        
        # Use thread pool for I/O operations
        loop = asyncio.get_event_loop()
        tensor = await loop.run_in_executor(
            self._thread_pool, 
            lambda: torch.load(path, map_location="cpu") * weight
        )
        
        # Cache the loaded tensor (limit cache size)
        if len(self._tensor_pool) < 50:  # Prevent unlimited growth
            self._tensor_pool[cache_key] = tensor
        
        return tensor

    async def _get_voices_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path with enhanced caching."""
        try:
            # Check path cache first
            if voice in self._path_cache:
                return voice, self._path_cache[voice]
            
            split_voice = re.split(r"([-+])", voice)

            # Single voice optimization
            if len(split_voice) == 1:
                if ("(" not in voice and ")" not in voice) or settings.voice_weight_normalization:
                    async with self._io_semaphore:
                        path = await self._voice_manager.get_voice_path(voice)
                        if not path:
                            raise RuntimeError(f"Voice not found: {voice}")
                        logger.debug(f"Using single voice path: {path}")
                        # Cache the path
                        self._path_cache[voice] = path
                        return voice, path

            # Combined voice processing with caching
            cache_key = f"combined_{voice}"
            if cache_key in self._voice_cache:
                return voice, self._voice_cache[cache_key]

            total_weight = 0
            for voice_index in range(0, len(split_voice), 2):
                voice_object = split_voice[voice_index]
                if "(" in voice_object and ")" in voice_object:
                    voice_name = voice_object.split("(")[0].strip()
                    voice_weight = float(voice_object.split("(")[1].split(")")[0])
                else:
                    voice_name = voice_object
                    voice_weight = 1
                total_weight += voice_weight
                split_voice[voice_index] = (voice_name, voice_weight)

            if not settings.voice_weight_normalization:
                total_weight = 1

            # Load first voice
            path = await self._voice_manager.get_voice_path(split_voice[0][0])
            combined_tensor = await self._load_voice_from_path(path, split_voice[0][1] / total_weight)

            # Process operations sequentially for stability
            for operation_index in range(1, len(split_voice) - 1, 2):
                path = await self._voice_manager.get_voice_path(split_voice[operation_index + 1][0])
                voice_tensor = await self._load_voice_from_path(
                    path, split_voice[operation_index + 1][1] / total_weight
                )
                
                if split_voice[operation_index] == "+":
                    combined_tensor += voice_tensor
                else:
                    combined_tensor -= voice_tensor

            # Save and cache combined voice
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, f"{voice}.pt")
            logger.debug(f"Saving combined voice to: {combined_path}")
            
            # Use thread pool for I/O operations
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._thread_pool, 
                lambda: torch.save(combined_tensor, combined_path)
            )
            
            # Cache the combined voice path
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
        """Generate and stream audio chunks with performance optimizations."""
        stream_normalizer = AudioNormalizer()
        chunk_index = 0
        current_offset = 0.0
        
        try:
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voices_path(voice)
            logger.debug(f"Using voice path: {voice_path}")

            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream")

            # Process text in chunks with smart splitting (PRESERVED ORIGINAL FLOW)
            async for chunk_text, tokens in smart_split(
                text, lang_code=pipeline_lang_code, normalization_options=normalization_options,
            ):
                try:
                    # Process audio for chunk
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
                except Exception as e:
                    logger.error(f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}")
                    continue

            # Finalize stream (PRESERVED ORIGINAL LOGIC)
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
            voice_name, voice_path = await self._get_voices_path(voice)

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