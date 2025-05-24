"""TTS service using model and voice managers."""

import asyncio
import os
import re
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import gc
from functools import lru_cache

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
    """Ultra-fast TTS service."""

    # MAXIMUM concurrency for sub-second latency
    _chunk_semaphore = asyncio.Semaphore(96)   # 96 concurrent chunks!
    _io_semaphore = asyncio.Semaphore(48)      # 48 I/O operations
    _generation_semaphore = asyncio.Semaphore(16)  # Model generation limit
    
    # Massive caching
    _voice_cache = {}
    _path_cache = {}
    _tensor_pool = {}
    _text_cache = {}  # Cache processed text chunks
    _normalizer_pool = []  # Pool of pre-created normalizers
    
    # Larger thread pools
    _thread_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tts-cpu")
    _io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tts-io")
    
    # Hot path optimizations
    _last_voice_tensor = None
    _last_voice_name = None
    _compiled_patterns = {}
    
    # Pipeline warming
    _warmed_pipelines = set()

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None
        self._setup_optimizations()

    def _setup_optimizations(self):
        """Setup micro-optimizations."""
        # Pre-compile all regex patterns
        self._compiled_patterns = {
            'voice_split': re.compile(r"([-+])"),
            'voice_weight': re.compile(r"(.+?)\(([0-9.]+)\)"),
            'whitespace': re.compile(r'\s+'),
        }
        
        # Pre-create normalizer pool
        for _ in range(10):
            self._normalizer_pool.append(AudioNormalizer())

    @classmethod
    async def create(cls, output_dir: str = None) -> "TTSService":
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        # AGGRESSIVE pre-warming
        await service._ultra_prewarm()
        return service

    async def _ultra_prewarm(self):
        """Ultra-aggressive system pre-warming for sub-second latency."""
        logger.info("Starting ultra pre-warming...")
        start_time = time.time()
        
        try:
            # Pre-warm ALL common voices in parallel
            all_voices = [
                settings.default_voice, "af_bella", "af_sarah", "af_nicole", "af_heart",
                "af_alloy", "af_aoede", "af_jadzia", "af_jessica", "af_kore", "af_nova",
                "af_river", "af_sky", "am_adam", "am_echo", "am_eric", "am_liam", 
                "am_michael", "am_onyx", "bm_daniel", "bm_george", "bf_emma", "bf_lily"
            ]
            
            backend = self.model_manager.get_backend()
            
            # Create all prewarm tasks
            tasks = []
            for voice_name in all_voices:
                task = asyncio.create_task(self._prewarm_voice_ultra(voice_name, backend))
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception))
            
            # Pre-warm pipelines for common languages
            common_langs = ['a', 'b', 'e', 'z', 'j']  # af, bm, em, zm, jf
            pipeline_tasks = []
            for lang in common_langs:
                task = asyncio.create_task(self._prewarm_pipeline(lang, backend))
                pipeline_tasks.append(task)
            
            await asyncio.gather(*pipeline_tasks, return_exceptions=True)
            
            # Pre-process common phrases for caching
            common_texts = [
                "Hello!", "How can I help you?", "Thank you!", "Yes", "No",
                "Please wait.", "One moment.", "I understand."
            ]
            
            text_tasks = []
            for text in common_texts:
                task = asyncio.create_task(self._cache_text_processing(text))
                text_tasks.append(task)
            
            await asyncio.gather(*text_tasks, return_exceptions=True)
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Ultra pre-warming completed in {elapsed:.1f}ms - {successful} voices, {len(self._warmed_pipelines)} pipelines, {len(self._text_cache)} texts cached")
            
        except Exception as e:
            logger.error(f"Ultra pre-warming failed: {e}")

    async def _prewarm_voice_ultra(self, voice_name: str, backend):
        """Ultra-fast voice pre-warming."""
        try:
            cache_key = f"{voice_name}_{backend.device}"
            if cache_key in self._voice_cache:
                return
                
            # Load voice and path in parallel
            voice_task = asyncio.create_task(
                self._voice_manager.load_voice(voice_name, device=backend.device)
            )
            path_task = asyncio.create_task(
                self._voice_manager.get_voice_path(voice_name)
            )
            
            voice_tensor, path = await asyncio.gather(voice_task, path_task, return_exceptions=True)
            
            if not isinstance(voice_tensor, Exception):
                self._voice_cache[cache_key] = voice_tensor
            if not isinstance(path, Exception):
                self._path_cache[voice_name] = path
                
            logger.debug(f"Ultra pre-warmed: {voice_name}")
        except Exception as e:
            logger.debug(f"Could not pre-warm {voice_name}: {e}")

    async def _prewarm_pipeline(self, lang_code: str, backend):
        """Pre-warm pipeline for language."""
        try:
            if isinstance(backend, KokoroV1):
                # Trigger pipeline creation with minimal text
                pipeline = backend._get_pipeline(lang_code)
                self._warmed_pipelines.add(lang_code)
                logger.debug(f"Pre-warmed pipeline for: {lang_code}")
        except Exception as e:
            logger.debug(f"Could not pre-warm pipeline {lang_code}: {e}")

    async def _cache_text_processing(self, text: str):
        """Pre-cache text processing results."""
        try:
            cache_key = f"text_{hash(text)}"
            if cache_key not in self._text_cache:
                # Process text through smart_split
                chunks = []
                async for chunk_text, tokens in smart_split(text, lang_code='a'):
                    chunks.append((chunk_text, tokens))
                self._text_cache[cache_key] = chunks
        except Exception as e:
            logger.debug(f"Could not cache text {text}: {e}")

    def _get_normalizer(self):
        """Get normalizer from pool or create new one."""
        if self._normalizer_pool:
            return self._normalizer_pool.pop()
        return AudioNormalizer()

    def _return_normalizer(self, normalizer):
        """Return normalizer to pool."""
        if len(self._normalizer_pool) < 20:
            self._normalizer_pool.append(normalizer)

    async def _process_chunk_ultra(
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
        """Ultra-fast chunk processing."""
        # Use different semaphore limits based on operation type
        semaphore = self._generation_semaphore if chunk_text else self._chunk_semaphore
        
        async with semaphore:
            try:
                # Fast finalization path
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

                # Skip empty chunks immediately
                if not tokens and not chunk_text:
                    return

                backend = self.model_manager.get_backend()

                if isinstance(backend, KokoroV1):
                    # Hot path: use cached voice tensor if same voice
                    if voice_name == self._last_voice_name and self._last_voice_tensor:
                        voice_path = self._last_voice_tensor
                    else:
                        self._last_voice_name = voice_name
                        self._last_voice_tensor = voice_path
                    
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        if output_format:
                            try:
                                # Process audio immediately to reduce latency
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data, output_format, writer, speed, chunk_text,
                                    is_last_chunk=is_last, normalizer=normalizer,
                                )
                                yield chunk_data
                            except Exception as e:
                                logger.error(f"Audio conversion failed: {str(e)}")
                        else:
                            chunk_data = AudioService.trim_audio(
                                chunk_data, chunk_text, speed, is_last, normalizer
                            )
                            yield chunk_data
                else:
                    # Ultra-fast voice lookup
                    cache_key = f"{voice_name}_{backend.device}"
                    voice_tensor = self._voice_cache.get(cache_key)
                    
                    if voice_tensor is None:
                        # Emergency load - should rarely happen with pre-warming
                        async with self._io_semaphore:
                            voice_tensor = await self._voice_manager.load_voice(
                                voice_name, device=backend.device
                            )
                            self._voice_cache[cache_key] = voice_tensor
                    
                    chunk_data = await self.model_manager.generate(
                        tokens, voice_tensor, speed=speed, return_timestamps=return_timestamps,
                    )

                    if chunk_data.audio is None or len(chunk_data.audio) == 0:
                        return

                    if output_format:
                        try:
                            chunk_data = await AudioService.convert_audio(
                                chunk_data, output_format, writer, speed, chunk_text,
                                normalizer=normalizer, is_last_chunk=is_last,
                            )
                            yield chunk_data
                        except Exception as e:
                            logger.error(f"Audio conversion failed: {str(e)}")
                    else:
                        yield AudioService.trim_audio(
                            chunk_data, chunk_text, speed, is_last, normalizer
                        )
            except Exception as e:
                logger.error(f"Chunk processing failed: {str(e)}")

    @lru_cache(maxsize=1000)
    def _parse_voice_fast(self, voice: str) -> tuple:
        """Ultra-fast voice parsing with LRU cache."""
        split_voice = self._compiled_patterns['voice_split'].split(voice)
        
        if len(split_voice) == 1:
            return (voice, None)  # Single voice
        
        # Parse combined voice
        parsed = []
        total_weight = 0
        
        for i in range(0, len(split_voice), 2):
            voice_obj = split_voice[i]
            match = self._compiled_patterns['voice_weight'].match(voice_obj)
            
            if match:
                name, weight = match.groups()
                weight = float(weight)
            else:
                name = voice_obj
                weight = 1.0
            
            parsed.append((name, weight))
            total_weight += weight
        
        return (voice, parsed, total_weight)

    async def _get_voices_path_ultra(self, voice: str) -> Tuple[str, str]:
        """Ultra-fast voice path resolution."""
        # Check cache first
        if voice in self._path_cache:
            return voice, self._path_cache[voice]
        
        # Parse voice with cached function
        parse_result = self._parse_voice_fast(voice)
        
        if parse_result[1] is None:  # Single voice
            if voice in self._path_cache:
                return voice, self._path_cache[voice]
            
            # Emergency load - should be rare with pre-warming
            path = await self._voice_manager.get_voice_path(voice)
            if not path:
                raise RuntimeError(f"Voice not found: {voice}")
            
            self._path_cache[voice] = path
            return voice, path
        
        # Combined voice
        voice_name, parsed_voices, total_weight = parse_result
        cache_key = f"combined_{voice}"
        
        if cache_key in self._voice_cache:
            return voice, self._voice_cache[cache_key]
        
        # Load and combine voices
        if not settings.voice_weight_normalization:
            total_weight = 1
        
        # Load first voice
        first_name, first_weight = parsed_voices[0]
        if first_name not in self._path_cache:
            path = await self._voice_manager.get_voice_path(first_name)
            self._path_cache[first_name] = path
        
        combined_tensor = await self._load_tensor_fast(
            self._path_cache[first_name], first_weight / total_weight
        )
        
        # Process remaining voices
        for i in range(1, len(parsed_voices)):
            if i-1 < len(parsed_voices) - 1:  # Check for operator
                op = "+"  # Default to addition
                voice_name, weight = parsed_voices[i]
                
                if voice_name not in self._path_cache:
                    path = await self._voice_manager.get_voice_path(voice_name)
                    self._path_cache[voice_name] = path
                
                tensor = await self._load_tensor_fast(
                    self._path_cache[voice_name], weight / total_weight
                )
                
                if op == "+":
                    combined_tensor += tensor
                else:
                    combined_tensor -= tensor
        
        # Save combined voice
        temp_path = os.path.join(tempfile.gettempdir(), f"voice_{abs(hash(voice))}.pt")
        
        # Use fast I/O pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._io_pool,
            lambda: torch.save(combined_tensor, temp_path)
        )
        
        self._voice_cache[cache_key] = temp_path
        self._path_cache[voice] = temp_path
        
        return voice, temp_path

    async def _load_tensor_fast(self, path: str, weight: float):
        """Ultra-fast tensor loading."""
        cache_key = f"tensor_{path}_{weight}"
        if cache_key in self._tensor_pool:
            return self._tensor_pool[cache_key]
        
        loop = asyncio.get_event_loop()
        tensor = await loop.run_in_executor(
            self._io_pool,
            lambda: torch.load(path, map_location="cpu", weights_only=True) * weight
        )
        
        # Aggressive caching
        if len(self._tensor_pool) < 200:
            self._tensor_pool[cache_key] = tensor
        
        return tensor

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
        """Ultra-fast audio streaming - target <500ms latency."""
        # Get normalizer from pool
        stream_normalizer = self._get_normalizer()
        
        try:
            # Fast path lookups
            voice_name, voice_path = await self._get_voices_path_ultra(voice)
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            
            # Check if text is cached
            cache_key = f"text_{hash(text)}_{pipeline_lang_code}"
            if cache_key in self._text_cache:
                chunks = self._text_cache[cache_key]
            else:
                # Process normally but cache result
                chunks = []
                async for chunk_text, tokens in smart_split(
                    text, lang_code=pipeline_lang_code, normalization_options=normalization_options,
                ):
                    chunks.append((chunk_text, tokens))
                
                # Cache for future use
                if len(self._text_cache) < 1000:
                    self._text_cache[cache_key] = chunks

            # Process all chunks with maximum concurrency
            chunk_index = 0
            current_offset = 0.0
            
            # Create tasks for ALL chunks immediately
            tasks = []
            for chunk_text, tokens in chunks:
                task = asyncio.create_task(self._process_chunk_ultra(
                    chunk_text, tokens, voice_name, voice_path, speed, writer,
                    output_format, is_first=(chunk_index == 0), is_last=False,
                    normalizer=stream_normalizer, lang_code=pipeline_lang_code,
                    return_timestamps=return_timestamps,
                ))
                tasks.append(task)
                chunk_index += 1
            
            # Process results as they complete
            for task in asyncio.as_completed(tasks):
                try:
                    async for chunk_data in await task:
                        if chunk_data.word_timestamps is not None:
                            for timestamp in chunk_data.word_timestamps:
                                timestamp.start_time += current_offset
                                timestamp.end_time += current_offset

                        current_offset += len(chunk_data.audio) / 24000

                        if chunk_data.output is not None:
                            yield chunk_data
                        
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")

            # Finalize
            if chunk_index > 0:
                async for chunk_data in self._process_chunk_ultra(
                    "", [], voice_name, voice_path, speed, writer, output_format,
                    is_first=False, is_last=True, normalizer=stream_normalizer,
                    lang_code=pipeline_lang_code,
                ):
                    if chunk_data.output is not None:
                        yield chunk_data

        finally:
            # Return normalizer to pool
            self._return_normalizer(stream_normalizer)

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
            voice_name, voice_path = await self._get_voices_path_ultra(voice)

            if isinstance(backend, KokoroV1):
                result = None
                pipeline_lang_code = lang_code if lang_code else voice[:1].lower()

                for r in backend._get_pipeline(pipeline_lang_code).generate_from_tokens(
                    tokens=phonemes, voice=voice_path, speed=speed,
                ):
                    if r.audio is not None:
                        result = r
                        break

                if result is None or result.audio is None:
                    raise ValueError("No audio generated")

                processing_time = time.time() - start_time
                return result.audio.numpy(), processing_time
            else:
                raise ValueError("Phoneme generation only supported with Kokoro V1 backend")

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise