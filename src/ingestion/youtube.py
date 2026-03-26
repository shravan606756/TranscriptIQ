import yt_dlp
import os
import tempfile
import re
from config import AUDIO_CACHE_DIR

def fetch_youtube_transcript(url):
    """
    Attempt to extract subtitles/captions using yt-dlp.
    Tries in order: manual English subs -> auto-generated English -> any English variant.
    
    Returns:
        tuple: (transcript_text, source_type) or (None, None) if failed
        source_type: 'manual', 'auto-generated', or None
    """
    
    # Use a temporary directory for subtitle files
    with tempfile.TemporaryDirectory() as temp_dir:
        subtitle_path = os.path.join(temp_dir, 'subtitle')
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'subtitlesformat': 'vtt',  # VTT is simpler to parse than json3
            'outtmpl': subtitle_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check what subtitles are available
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # Priority: manual subs > auto-generated subs
                selected_lang = None
                is_auto = False
                
                # Try manual English subtitles first
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in subtitles:
                        selected_lang = lang
                        is_auto = False
                        break
                
                # Fallback to auto-generated
                if not selected_lang:
                    for lang in ['en', 'en-US', 'en-GB']:
                        if lang in automatic_captions:
                            selected_lang = lang
                            is_auto = True
                            break
                
                if not selected_lang:
                    return None, None
                
                # Download the subtitle file
                ydl_opts['subtitleslangs'] = [selected_lang]
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])
                
                # Find and read the subtitle file
                subtitle_file = None
                for file in os.listdir(temp_dir):
                    if file.endswith('.vtt'):
                        subtitle_file = os.path.join(temp_dir, file)
                        break
                
                if not subtitle_file or not os.path.exists(subtitle_file):
                    return None, None
                
                # Parse VTT file
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    vtt_content = f.read()
                
                # Extract text from VTT (skip WEBVTT header and timestamps)
                text_parts = []
                for line in vtt_content.split('\n'):
                    line = line.strip()
                    # Skip WEBVTT header, timestamps, and empty lines
                    if (line and 
                        not line.startswith('WEBVTT') and 
                        not line.startswith('Kind:') and
                        not line.startswith('Language:') and
                        '-->' not in line and
                        not line.isdigit()):
                        
                        # CRITICAL FIX: Strip inline VTT tags (<c>, timestamps, etc.)
                        cleaned = re.sub(r'<[^>]+>', '', line)
                        cleaned = cleaned.replace('\n', ' ').strip()
                        
                        if cleaned and cleaned not in text_parts:  # Avoid duplicates
                            text_parts.append(cleaned)
                
                full_text = ' '.join(text_parts)
                full_text = ' '.join(full_text.split())  # Normalize whitespace
                
                source_type = 'auto-generated' if is_auto else 'manual'
                
                return full_text if full_text else None, source_type
                
        except Exception as e:
            print(f"Subtitle extraction failed: {e}")
            return None, None


def download_audio(url):
    """
    Download audio from YouTube video.
    
    Returns:
        str: Path to downloaded audio file
    """
    output_path = f"{AUDIO_CACHE_DIR}/yt_audio.%(ext)s"
    
    # Ensure directory exists
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    
    return filename


def get_video_info(url):
    """
    Get basic video information without downloading.
    Useful for debugging and showing video details to user.
    
    Returns:
        dict: Video metadata (title, duration, has_subs, etc.)
    """
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Check for subtitle availability
            has_manual = False
            has_auto = False
            
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subtitles:
                    has_manual = True
                if lang in automatic_captions:
                    has_auto = True
            
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'channel': info.get('channel'),
                'has_manual_subs': has_manual,
                'has_auto_subs': has_auto,
            }
    except Exception as e:
        print(f"Failed to get video info: {e}")
        return None