import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.youtube import fetch_youtube_transcript, get_video_info

@patch('src.ingestion.youtube.yt_dlp.YoutubeDL')
def test_get_video_info(mock_ytdl):
    mock_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        'title': 'Test Video',
        'duration': 120,
        'channel': 'Test Channel',
        'subtitles': {'en': {}},
        'automatic_captions': {}
    }

    info = get_video_info('http://test.url')
    assert info is not None
    assert info['title'] == 'Test Video'
    assert info['has_manual_subs'] == True
    assert info['has_auto_subs'] == False

@patch('src.ingestion.youtube.tempfile.TemporaryDirectory')
@patch('src.ingestion.youtube.yt_dlp.YoutubeDL')
def test_fetch_youtube_transcript_no_subs(mock_ytdl, mock_temp):
    mock_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        'subtitles': {},
        'automatic_captions': {}
    }

    transcript, source = fetch_youtube_transcript('http://test.url')
    assert transcript is None
    assert source is None
