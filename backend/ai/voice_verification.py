import os
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

VOICE_DIR = "voice_samples"
os.makedirs(VOICE_DIR, exist_ok=True)

MODEL_NAME = "microsoft/wavlm-base-plus-sv"
VOICE_MATCH_THRESHOLD = 0.88

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMForXVector.from_pretrained(MODEL_NAME)
model.eval()


def ensure_wav(audio_path: str) -> str:
    ext = os.path.splitext(audio_path)[1].lower()

    if ext == ".wav":
        return audio_path

    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    return wav_path


def load_audio(audio_path: str):
    wav_path = ensure_wav(audio_path)

    audio, sample_rate = sf.read(wav_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000
        )

    return audio


def get_embedding(audio):
    if isinstance(audio, str):
        audio = load_audio(audio)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.embeddings

    embedding = F.normalize(embedding, dim=-1)

    return embedding


def get_segment_embeddings(audio, segment_len=48000, overlap=24000):
    if len(audio) < segment_len:
        return [get_embedding(audio)]
    embeddings = []
    step = segment_len - overlap
    for start in range(0, len(audio) - segment_len + 1, step):
        segment = audio[start : start + segment_len]
        embeddings.append(get_embedding(segment))
    if not embeddings:
        embeddings.append(get_embedding(audio))
    return embeddings


def verify_voice(
    user_id: str,
    verification_audio_path: str,
    threshold: float = VOICE_MATCH_THRESHOLD
):
    enrolled_wav = os.path.join(VOICE_DIR, f"{user_id}.wav")
    enrolled_webm = os.path.join(VOICE_DIR, f"{user_id}.webm")
    enrolled_mp3 = os.path.join(VOICE_DIR, f"{user_id}.mp3")

    if os.path.exists(enrolled_wav):
        enrolled_path = enrolled_wav
    elif os.path.exists(enrolled_webm):
        enrolled_path = enrolled_webm
    elif os.path.exists(enrolled_mp3):
        enrolled_path = enrolled_mp3
    else:
        return {
            "success": False,
            "match": False,
            "similarity": 0.0,
            "threshold": threshold,
            "message": "Enrolled voice sample not found"
        }

    try:
        # Load enrolled and verification audios
        enrolled_audio = load_audio(enrolled_path)
        verification_audio = load_audio(verification_audio_path)

        # 1. Validate duration (16000 Hz sample rate)
        enrolled_duration = len(enrolled_audio) / 16000.0
        verification_duration = len(verification_audio) / 16000.0

        if enrolled_duration < 3.0 or verification_duration < 3.0:
            return {
                "success": False,
                "match": False,
                "similarity": 0.0,
                "threshold": threshold,
                "message": "Voice sample too short (min 3s required)"
            }

        # 2. Validate energy (RMS) to detect quiet/silent audio
        enrolled_rms = np.sqrt(np.mean(enrolled_audio ** 2))
        verification_rms = np.sqrt(np.mean(verification_audio ** 2))

        # Quiet threshold set at 0.002
        if enrolled_rms < 0.002 or verification_rms < 0.002:
            return {
                "success": False,
                "match": False,
                "similarity": 0.0,
                "threshold": threshold,
                "message": "Voice sample too quiet / silent"
            }

        # Extract segment embeddings
        e_embs = get_segment_embeddings(enrolled_audio)
        v_embs = get_segment_embeddings(verification_audio)

        # Compute cosine similarity between all segment combinations and average them
        similarities = []
        for e_emb in e_embs:
            for v_emb in v_embs:
                sim = F.cosine_similarity(e_emb, v_emb).item()
                similarities.append(sim)

        final_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Match decision
        match = final_similarity >= threshold

        # Detailed logging
        print("=" * 50)
        print("VOICE VERIFICATION")
        print(f"User ID: {user_id}")
        print(f"Similarity: {final_similarity:.4f}")
        print(f"Threshold: {threshold}")
        print(f"Match: {match}")
        print("=" * 50)

        return {
            "success": True,
            "match": match,
            "similarity": round(final_similarity, 4),
            "threshold": threshold,
            "message": "Voice matched" if match else "Voice verification failed"
        }

    except Exception as e:
        return {
            "success": False,
            "match": False,
            "similarity": 0.0,
            "threshold": threshold,
            "message": f"Voice verification error: {str(e)}"
        }