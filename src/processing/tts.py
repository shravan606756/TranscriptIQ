import tempfile
import os
from gtts import gTTS

def generate_tts_audio(text, lang='en', slow=False):
    """
    Generate Text-to-Speech audio from text and return raw bytes.
    
    Args:
        text (str): The text to synthesize
        lang (str): Language code (default: 'en')
        slow (bool): Whether to read slowly (default: False)
        
    Returns:
        bytes: Raw MP3 audio bytes
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text")
        
    # Generate speech
    tts = gTTS(text=text, lang=lang, slow=slow)
    
    # Save to temp file and read bytes
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            audio_path = fp.name
            
        tts.save(audio_path)
        
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            
        return audio_bytes
    finally:
        # Guarantee cleanup
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
