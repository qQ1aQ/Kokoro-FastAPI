"""TTS service using model and voice managers."""

import asyncio
import os
import re
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union

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
    """Text-to-speech service optimized for high GPU utilization."""

    # Significantly increase concurrent chunk processing for GPU utilization
    # RunPod GPUs can handle much more parallel processing
    _chunk_semaphore = asyncio.Semaphore(32)  # Increased from 4 to 32
    
    # Add batch processing semaphore for multiple requests
    _batch_semaphore = asyncio.Semaphore(8)  # Allow 8 concurrent batch requests
    
    # Voice cache to avoid repeated loading
    _voice_cache = {}
    _voice_cache_lock = asyncio.Lock()

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
        return service

    async def _process_chunk_batch(
        self,
        chunks: List[Tuple[str, List[int]]],
        voice_name: str,
        voice_path: str,
        speed: float,
        writer: StreamingAudioWriter,
        output_format: Optional[str] = None,
        normalizer: Optional[AudioNormalizer] = None,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[List[AudioChunk], None]:
        """Process multiple chunks in parallel for maximum GPU utilization."""
        async with self._batch_semaphore:
            try:
                # Process chunks in parallel using asyncio.gather
                tasks = []
                for i, (chunk_text, tokens) in enumerate(chunks):
                    task = self._process_chunk(
                        chunk_text,
                        tokens,
                        voice_name,
                        voice_path,
                        speed,
                        writer,
                        output_format,
                        is_first=(i == 0),
                        is_last=(i == len(chunks) - 1),
                        normalizer=normalizer,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    )
                    tasks.append(task)
                
                # Execute all chunks in parallel
                batch_results = []
                async for chunk_results in asyncio.as_completed(tasks):
                    chunk_data_list = []
                    async for chunk_data in chunk_results:
                        chunk_data_list.append(chunk_data)
                    batch_results.append(chunk_data_list)
                
                yield batch_results
                
            except Exception as e:
                logger.error(f"Failed to process chunk batch: {str(e)}")

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
        """Process tokens into audio with GPU optimization."""
        # Reduce semaphore contention for better GPU utilization
        async with self._chunk_semaphore:
            try:
                # Handle stream finalization
                if is_last:
                    # Skip format conversion for raw audio mode
                    if not output_format:
                        yield AudioChunk(np.array([], dtype=np.int16), output=b"")
                        return
                    chunk_data = await AudioService.convert_audio(
                        AudioChunk(
                            np.array([], dtype=np.float32)
                        ),  # Dummy data for type checking
                        output_format,
                        writer,
                        speed,
                        "",
                        normalizer=normalizer,
                        is_last_chunk=True,
                    )
                    yield chunk_data
                    return

                # Skip empty chunks
                if not tokens and not chunk_text:
                    return

                # Get backend
                backend = self.model_manager.get_backend()

                # Generate audio using pre-warmed model
                if isinstance(backend, KokoroV1):
                    chunk_index = 0
                    # For Kokoro V1, pass text and voice info with lang_code
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        # For streaming, convert to bytes
                        if output_format:
                            try:
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data,
                                    output_format,
                                    writer,
                                    speed,
                                    chunk_text,
                                    is_last_chunk=is_last,
                                    normalizer=normalizer,
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
                    # For legacy backends, load voice tensor with caching
                    voice_tensor = await self._get_cached_voice_tensor(
                        voice_name, backend.device
                    )
                    chunk_data = await self.model_manager.generate(
                        tokens,
                        voice_tensor,
                        speed=speed,
                        return_timestamps=return_timestamps,
                    )

                    if chunk_data.audio is None:
                        logger.error("Model generated None for audio chunk")
                        return

                    if len(chunk_data.audio) == 0:
                        logger.error("Model generated empty audio chunk")
                        return

                    # For streaming, convert to bytes
                    if output_format:
                        try:
                            chunk_data = await AudioService.convert_audio(
                                chunk_data,
                                output_format,
                                writer,
                                speed,
                                chunk_text,
                                normalizer=normalizer,
                                is_last_chunk=is_last,
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

    async def _get_cached_voice_tensor(self, voice_name: str, device: str) -> torch.Tensor:
        """Get cached voice tensor to avoid repeated loading."""
        cache_key = f"{voice_name}_{device}"
        
        async with self._voice_cache_lock:
            if cache_key not in self._voice_cache:
                logger.debug(f"Loading and caching voice tensor: {voice_name}")
                voice_tensor = await self._voice_manager.load_voice(voice_name, device=device)
                self._voice_cache[cache_key] = voice_tensor
                # Limit cache size to prevent memory issues
                if len(self._voice_cache) > 20:  # Keep only 20 most recent voices
                    oldest_key = next(iter(self._voice_cache))
                    del self._voice_cache[oldest_key]
            else:
                logger.debug(f"Using cached voice tensor: {voice_name}")
            
            return self._voice_cache[cache_key]

    async def _load_voice_from_path(self, path: str, weight: float):
        # Check if the path is None and raise a ValueError if it is not
        if not path:
            raise ValueError(f"Voice not found at path: {path}")

        logger.debug(f"Loading voice tensor from path: {path}")
        return torch.load(path, map_location="cpu") * weight

    async def _get_voices_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path, handling combined voices with GPU optimization.

        Args:
            voice: Voice name or combined voice names (e.g., 'af_jadzia+af_jessica')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Split the voice on + and - and ensure that they get added to the list eg: hi+bob = ["hi","+","bob"]
            split_voice = re.split(r"([-+])", voice)

            # If it is only once voice there is no point in loading it up, doing nothing with it, then saving it
            if len(split_voice) == 1:
                # Since its a single voice the only time that the weight would matter is if voice_weight_normalization is off
                if (
                    "(" not in voice and ")" not in voice
                ) or settings.voice_weight_normalization == True:
                    path = await self._voice_manager.get_voice_path(voice)
                    if not path:
                        raise RuntimeError(f"Voice not found: {voice}")
                    logger.debug(f"Using single voice path: {path}")
                    return voice, path

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

            # If voice_weight_normalization is false prevent normalizing the weights by setting the total_weight to 1 so it divides each weight by 1
            if settings.voice_weight_normalization == False:
                total_weight = 1

            # Load the first voice as the starting point for voices to be combined onto - optimize for GPU
            path = await self._voice_manager.get_voice_path(split_voice[0][0])
            combined_tensor = await self._load_voice_from_path(
                path, split_voice[0][1] / total_weight
            )
            
            # Move to GPU immediately for faster operations
            device = self.model_manager.get_backend().device
            if device != "cpu":
                combined_tensor = combined_tensor.to(device)

            # Loop through each + or - in split_voice so they can be applied to combined voice
            for operation_index in range(1, len(split_voice) - 1, 2):
                # Get the voice path of the voice 1 index ahead of the operator
                path = await self._voice_manager.get_voice_path(
                    split_voice[operation_index + 1][0]
                )
                voice_tensor = await self._load_voice_from_path(
                    path, split_voice[operation_index + 1][1] / total_weight
                )
                
                # Move to same device for faster operations
                if device != "cpu":
                    voice_tensor = voice_tensor.to(device)

                # Either add or subtract the voice from the current combined voice
                if split_voice[operation_index] == "+":
                    combined_tensor += voice_tensor
                else:
                    combined_tensor -= voice_tensor

            # Save the new combined voice so it can be loaded latter
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, f"{voice}.pt")
            logger.debug(f"Saving combined voice to: {combined_path}")
            # Save on CPU for portability
            torch.save(combined_tensor.cpu(), combined_path)
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
        """Generate and stream audio chunks with optimized GPU utilization."""
        stream_normalizer = AudioNormalizer()
        chunk_index = 0
        current_offset = 0.0
        try:
            # Get backend
            backend = self.model_manager.get_backend()

            # Get voice path, handling combined voices
            voice_name, voice_path = await self._get_voices_path(voice)
            logger.debug(f"Using voice path: {voice_path}")

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream"
            )

            # Collect chunks for batch processing
            chunks = []
            async for chunk_text, tokens in smart_split(
                text,
                lang_code=pipeline_lang_code,
                normalization_options=normalization_options,
            ):
                chunks.append((chunk_text, tokens))

            # Process chunks in batches for better GPU utilization
            batch_size = min(8, len(chunks))  # Process up to 8 chunks simultaneously
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                try:
                    # Process batch of chunks in parallel
                    tasks = []
                    for j, (chunk_text, tokens) in enumerate(batch_chunks):
                        task = self._process_chunk(
                            chunk_text,
                            tokens,
                            voice_name,
                            voice_path,
                            speed,
                            writer,
                            output_format,
                            is_first=(chunk_index == 0),
                            is_last=(i + j == len(chunks) - 1),
                            normalizer=stream_normalizer,
                            lang_code=pipeline_lang_code,
                            return_timestamps=return_timestamps,
                        )
                        tasks.append({'task': task, 'text': chunk_text})
                    
                    # Process tasks and yield results in order
                    for task_info in tasks:
                        try:
                            async for chunk_data in task_info['task']:
                                if chunk_data.word_timestamps is not None:
                                    for timestamp in chunk_data.word_timestamps:
                                        timestamp.start_time += current_offset
                                        timestamp.end_time += current_offset

                                current_offset += len(chunk_data.audio) / 24000

                                if chunk_data.output is not None:
                                    yield chunk_data
                                else:
                                    logger.warning(
                                        f"No audio generated for chunk: '{task_info['text'][:100]}...'"
                                    )
                                chunk_index += 1
                        except Exception as e:
                            logger.error(
                                f"Failed to process audio for chunk: '{task_info['text'][:100]}...'. Error: {str(e)}"
                            )
                            continue
                except Exception as e:
                    logger.error(f"Failed to process batch: {str(e)}")
                    continue

            # Only finalize if we successfully processed at least one chunk
            if chunk_index > 0:
                try:
                    # Empty tokens list to finalize audio
                    async for chunk_data in self._process_chunk(
                        "",  # Empty text
                        [],  # Empty tokens
                        voice_name,
                        voice_path,
                        speed,
                        writer,
                        output_format,
                        is_first=False,
                        is_last=True,  # Signal this is the last chunk
                        normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,  # Pass lang_code
                    ):
                        if chunk_data.output is not None:
                            yield chunk_data
                except Exception as e:
                    logger.error(f"Failed to finalize audio stream: {str(e)}")

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
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
                text,
                voice,
                writer,
                speed=speed,
                normalization_options=normalization_options,
                return_timestamps=return_timestamps,
                lang_code=lang_code,
                output_format=None,
            ):
                if len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)

            combined_audio_data = AudioChunk.combine(audio_data_chunks)
            return combined_audio_data
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> torch.Tensor:
        """Combine multiple voices.

        Returns:
            Combined voice tensor
        """

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
        """Generate audio directly from phonemes.

        Args:
            phonemes: Phonemes in Kokoro format
            voice: Voice name
            speed: Speed multiplier
            lang_code: Optional language code override

        Returns:
            Tuple of (audio array, processing time)
        """
        start_time = time.time()
        try:
            # Get backend and voice path
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voices_path(voice)

            if isinstance(backend, KokoroV1):
                # For Kokoro V1, use generate_from_tokens with raw phonemes
                result = None
                # Use provided lang_code or determine from voice name
                pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
                logger.info(
                    f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme pipeline"
                )

                try:
                    # Use backend's pipeline management
                    for r in backend._get_pipeline(
                        pipeline_lang_code
                    ).generate_from_tokens(
                        tokens=phonemes,  # Pass raw phonemes string
                        voice=voice_path,
                        speed=speed,
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
                raise ValueError(
                    "Phoneme generation only supported with Kokoro V1 backend"
                )

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise

    @classmethod
    def clear_voice_cache(cls):
        """Clear the voice tensor cache to free memory."""
        cls._voice_cache.clear()
        logger.info("Voice cache cleared")

    @classmethod
    def get_cache_stats(cls) -> dict:
        """Get voice cache statistics."""
        return {
            "cached_voices": len(cls._voice_cache),
            "cache_keys": list(cls._voice_cache.keys())
        }