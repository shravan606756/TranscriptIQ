import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#to ensure that the .env file is loaded
from dotenv import load_dotenv
load_dotenv()


import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.ingestion.youtube import get_video_info
from src.processing.summarize import summarize_text
from src.retrieval.rag import build_vector_store, generate_answer
from config import AUDIO_CACHE_DIR
from src.pipeline import process_youtube_pipeline, process_audio_pipeline
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
from src.processing.tts import generate_tts_audio
import base64

def update_session_state(text, summary, metrics, source, model="bart-large-cnn"):
    """Helper to commit a new transcript to session state and clear old caches."""
    st.session_state["transcript"] = text
    
    # Clear stale RAG data
    for key in ["rag_index", "rag_chunks", "qa_history"]:
        if key in st.session_state:
            del st.session_state[key]
            
    # Save isolated memory states for the active model
    st.session_state[f"summary_bart"] = summary
    st.session_state[f"summary_metrics_bart"] = metrics
    
    # Set active views
    st.session_state["summary"] = summary
    st.session_state["summary_metrics"] = metrics
    st.session_state["source"] = source
    st.session_state["current_model"] = model
    
    # Clear alternative model data when regenerating
    if "summary_t5" in st.session_state: del st.session_state["summary_t5"]
    if "summary_metrics_t5" in st.session_state: del st.session_state["summary_metrics_t5"]

# Page Config
st.set_page_config(
    page_title="Podcast Summarizer Pro",
    layout="wide"
)

# Custom CSS - FIXED: All text now visible in both light and dark modes
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Podcast Summarizer Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-Powered Transcript Extraction and Multi-Model Summarization</p>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Display Options")
    show_video_info = st.checkbox("Show video metadata", value=True)
    show_stats = st.checkbox("Show processing statistics", value=True)
    
    st.divider()
    
    st.subheader("Summary Configuration")
    detail_level = st.select_slider(
        "Detail level",
        options=["brief", "medium", "detailed"],
        value="medium",
        help="Brief: ~75% compression | Medium: ~70% compression | Detailed: ~65% compression"
    )
    
    st.divider()
    
    st.subheader("Model Information")
    
    with st.expander("BART-large-CNN (Detailed)", expanded=False):
        st.caption("**Processing Speed**: 20-40 seconds")
        st.caption("**Model Size**: 1.6 GB (406M parameters)")
        st.caption("**Quality**: Excellent for detailed summaries")
        st.caption("**Compression**: 60-75%")
    
    with st.expander("T5-base (Comparison)", expanded=False):
        st.caption("**Processing Speed**: 15-30 seconds")
        st.caption("**Model Size**: 900 MB (220M parameters)")
        st.caption("**Quality**: Good but aggressive compression")
        st.caption("**Compression**: 85-95%")
    
    st.divider()
    
    st.subheader("System Information")
    st.caption("Transcription: OpenAI Whisper (base)")
    st.caption("Summarization: BART-large-CNN / T5-base")
    st.caption("Video Extraction: yt-dlp")
    st.caption("Text-to-Speech: Google TTS")

# Main Layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Input")

    option = st.radio(
        "Select input type",
        ["YouTube Link", "Upload Audio"]
    )

    # -------------------- YOUTUBE INPUT --------------------
    if option == "YouTube Link":
        
        url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )

        if url.strip() and show_video_info:
            with st.spinner("Fetching video metadata..."):
                video_info = get_video_info(url)
            
            if video_info:
                st.markdown("**Video Information**")
                st.markdown(f'<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**Title**: {video_info['title']}")
                st.markdown(f"**Channel**: {video_info['channel']}")
                
                duration = video_info['duration']
                if duration:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    if hours > 0:
                        duration_str = f"{hours}h {minutes}m {seconds}s"
                    else:
                        duration_str = f"{minutes}m {seconds}s"
                    st.markdown(f"**Duration**: {duration_str}")
                
                if video_info['has_manual_subs']:
                    st.markdown("**Subtitles**: Manual captions available")
                elif video_info['has_auto_subs']:
                    st.markdown("**Subtitles**: Auto-generated captions available")
                else:
                    st.markdown("**Subtitles**: None (will use Whisper transcription)")
                
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Process", type="primary", use_container_width=True):

            if url.strip() == "":
                st.error("Please enter a valid URL")
            else:
                try:
                    status_placeholder = st.empty()
                    def ui_callback(msg):
                        status_placeholder.info(msg)
                        
                    # Process entire pipeline purely from facade
                    with st.spinner("Processing..."):
                        text, source, summary, metrics = process_youtube_pipeline(url, detail_level, ui_callback)
                    
                    status_placeholder.empty()
                    
                    if show_stats:
                        word_count = len(text.split())
                        st.markdown(f'<div class="stats-box">Transcript from {source}: {word_count:,} words</div>', unsafe_allow_html=True)
                        

                    # Use centralized commit function
                    update_session_state(text, summary, metrics, source)
                    st.success("Processing complete")
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)


    # -------------------- AUDIO UPLOAD --------------------
    elif option == "Upload Audio":

        uploaded_file = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "webm", "ogg"])

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f'<div class="stats-box">File: {uploaded_file.name} ({file_size_mb:.2f} MB)</div>', unsafe_allow_html=True)

            if st.button("Process", type="primary", use_container_width=True):
                try:
                    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
                    file_path = os.path.join(AUDIO_CACHE_DIR, uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.success("File uploaded successfully")

                    status_placeholder = st.empty()
                    def ui_callback(msg):
                        status_placeholder.info(msg)

                    # Process via facade
                    with st.spinner("Processing..."):
                        text, source, summary, metrics = process_audio_pipeline(file_path, detail_level, ui_callback)
                    
                    status_placeholder.empty()

                    if show_stats:
                        word_count = len(text.split())
                        st.markdown(f'<div class="stats-box">Transcript: {word_count:,} words</div>', unsafe_allow_html=True)

                    # Use centralized commit function
                    update_session_state(text, summary, metrics, source)
                    st.success("Processing complete")
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)

# -------------------- OUTPUT SECTION --------------------
with right_col:
    st.subheader("Results")

    if "transcript" in st.session_state:

        st.markdown(f'<div class="stats-box">Source: {st.session_state["source"]}</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Transcript", "Analytics", "Model Comparison", "Q&A"])

        with tab1:
            current_model = st.session_state.get("current_model", "bart-large-cnn")
            badge_class = "badge-bart" if current_model == "bart-large-cnn" else "badge-t5"
            
            st.markdown(f'<span class="model-badge {badge_class}">{current_model.upper()}</span>', unsafe_allow_html=True)
            st.markdown('<p class="section-header">Generated Summary</p>', unsafe_allow_html=True)
            
            summary_text = st.session_state["summary"]
            st.write(summary_text)
            
            if "summary_metrics" in st.session_state:
                metrics = st.session_state["summary_metrics"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", f"{metrics['summary_words']:,}")
                with col2:
                    st.metric("Compression", f"{metrics['compression_ratio']:.1f}%")
                with col3:
                    st.metric("Processing Time", f"{metrics['processing_time']:.1f}s")
            
            st.divider()
            
            # NEW: 3-column layout for buttons and audio
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="Download Summary",
                    data=summary_text,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # NEW: TTS Audio Generation
                if st.button("Generate Audio", use_container_width=True, type="secondary"):
                    with st.spinner("Generating audio..."):
                        try:
                            audio_bytes = generate_tts_audio(summary_text)
                            st.session_state[f"audio_{current_model}"] = audio_bytes
                            st.success("Audio generated")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Audio generation failed: {str(e)}")
            
            with col3:
                # Regenerate/Switch button
                if current_model == "bart-large-cnn":
                    t5_cached = "summary_t5" in st.session_state
                    btn_label = "Switch to T5-base" if t5_cached else "Regenerate with T5-base"
                    if st.button(btn_label, use_container_width=True, type="secondary"):
                        if t5_cached:
                            # Just switch — no re-inference
                            with st.spinner("Switching to T5-base..."):
                                st.session_state["summary"] = st.session_state["summary_t5"]
                                st.session_state["summary_metrics"] = st.session_state["summary_metrics_t5"]
                                st.session_state["current_model"] = "t5-base"
                            st.rerun()
                        else:
                            with st.spinner("Generating summary with T5-base..."):
                                summary_t5, metrics_t5 = summarize_text(
                                    st.session_state["transcript"],
                                    detail_level=detail_level,
                                    model_name="t5-base",
                                    return_metrics=True
                                )
                                st.session_state["summary_t5"] = summary_t5
                                st.session_state["summary_metrics_t5"] = metrics_t5
                                st.session_state["summary"] = summary_t5
                                st.session_state["summary_metrics"] = metrics_t5
                                st.session_state["current_model"] = "t5-base"
                            st.success("Summary generated with T5-base")
                            st.rerun()
                else:
                    if st.button("Switch to BART", use_container_width=True, type="secondary"):
                        with st.spinner("Switching to BART..."):
                            st.session_state["summary"] = st.session_state["summary_bart"]
                            st.session_state["summary_metrics"] = st.session_state["summary_metrics_bart"]
                            st.session_state["current_model"] = "bart-large-cnn"
                        st.rerun()
            
            # NEW: Audio Player (if audio exists for current model)
            audio_key = f"audio_{current_model}"
            if audio_key in st.session_state:
                st.markdown("**Audio Summary**")
                st.audio(st.session_state[audio_key], format='audio/mp3')
            
            # Show comparison notice
            if "summary_t5" in st.session_state and "summary_bart" in st.session_state and current_model == "bart-large-cnn":
                st.info("Both BART and T5 summaries available. View Model Comparison tab for analysis.")

        with tab2:
            st.markdown('<p class="section-header">Full Transcript</p>', unsafe_allow_html=True)
            transcript_text = st.session_state["transcript"]
            st.write(transcript_text)
            
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

        with tab3:
            st.markdown('<p class="section-header">Text Analytics</p>', unsafe_allow_html=True)
            
            transcript = st.session_state["transcript"]
            summary = st.session_state["summary"]
            
            trans_words = len(transcript.split())
            trans_chars = len(transcript)
            trans_sentences = transcript.count('.') + transcript.count('!') + transcript.count('?')
            
            summ_words = len(summary.split())
            summ_chars = len(summary)
            
            compression_ratio = (1 - summ_words / trans_words) * 100 if trans_words > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Transcript Metrics**")
                st.metric("Words", f"{trans_words:,}")
                st.metric("Characters", f"{trans_chars:,}")
                st.metric("Sentences", f"{trans_sentences:,}")
                avg_words = trans_words // trans_sentences if trans_sentences > 0 else 0
                st.metric("Avg words per sentence", f"{avg_words}")
            
            with col2:
                st.markdown("**Summary Metrics**")
                st.metric("Words", f"{summ_words:,}")
                st.metric("Characters", f"{summ_chars:,}")
                st.metric("Compression ratio", f"{compression_ratio:.1f}%")
                reading_time = max(1, (summ_words + 149) // 150)
                st.metric("Estimated reading time", f"{reading_time} min")
            
            st.markdown("**Length Comparison**")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Transcript', x=['Word Count'], y=[trans_words], marker_color='#1f77b4', 
                text=[f'{trans_words:,}'], textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Summary', x=['Word Count'], y=[summ_words], marker_color='#7b1fa2', 
                text=[f'{summ_words:,}'], textposition='auto'
            ))
            fig.update_layout(
                height=400, showlegend=True, yaxis_title="Words", 
                template="plotly_white", font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
            
            # CRITICAL FIX: Check for isolated memory states
            has_both = "summary_bart" in st.session_state and "summary_t5" in st.session_state
            
            if not has_both:
                st.info("Generate a T5-base summary to enable model comparison. Click 'Regenerate with T5-base' in the Summary tab.")
                
                st.markdown("### Expected Model Characteristics")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### BART-large-CNN")
                    st.markdown("- **Processing Speed**: 20-40 seconds\n- **Model Size**: 1.6 GB\n- **Compression**: 60-75%\n- **Quality**: Excellent for detailed summaries")
                with col2:
                    st.markdown("#### T5-base")
                    st.markdown("- **Processing Speed**: 15-30 seconds\n- **Model Size**: 900 MB\n- **Compression**: 85-95%\n- **Quality**: Good but aggressive compression")
            else:
                # CRITICAL FIX: Pull strictly from isolated memory states
                metrics_bart = st.session_state["summary_metrics_bart"]
                metrics_t5 = st.session_state["summary_metrics_t5"]
                
                st.markdown("### Summary Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### BART-large-CNN Summary")
                    st.markdown(f'<div class="stats-box">Words: {metrics_bart["summary_words"]:,} | '
                              f'Time: {metrics_bart["processing_time"]:.1f}s | '
                              f'Compression: {metrics_bart["compression_ratio"]:.1f}%</div>',
                              unsafe_allow_html=True)
                    st.write(st.session_state["summary_bart"])
                
                with col2:
                    st.markdown("#### T5-base Summary")
                    st.markdown(f'<div class="stats-box">Words: {metrics_t5["summary_words"]:,} | '
                              f'Time: {metrics_t5["processing_time"]:.1f}s | '
                              f'Compression: {metrics_t5["compression_ratio"]:.1f}%</div>',
                              unsafe_allow_html=True)
                    st.write(st.session_state["summary_t5"])
                
                st.divider()
                st.markdown("### Detailed Metrics Comparison")
                
                comp_df = pd.DataFrame({
                    'Metric': ['Summary Words', 'Processing Time (s)', 'Compression Ratio (%)', 'Chunks Processed'],
                    'BART-large-CNN': [
                        metrics_bart['summary_words'], 
                        round(metrics_bart['processing_time'], 1), 
                        round(metrics_bart['compression_ratio'], 1), 
                        metrics_bart['num_chunks']
                    ],
                    'T5-base': [
                        metrics_t5['summary_words'], 
                        round(metrics_t5['processing_time'], 1), 
                        round(metrics_t5['compression_ratio'], 1), 
                        metrics_t5['num_chunks']
                    ],
                    'Difference': [
                        metrics_bart['summary_words'] - metrics_t5['summary_words'],
                        round(metrics_bart['processing_time'] - metrics_t5['processing_time'], 1),
                        round(metrics_bart['compression_ratio'] - metrics_t5['compression_ratio'], 1),
                        metrics_bart['num_chunks'] - metrics_t5['num_chunks']
                    ]
                })
                
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_time = go.Figure()
                    fig_time.add_trace(go.Bar(
                        x=['BART-large-CNN', 'T5-base'], 
                        y=[metrics_bart['processing_time'], metrics_t5['processing_time']], 
                        marker_color=['#1f77b4', '#7b1fa2'],
                        text=[f"{metrics_bart['processing_time']:.1f}s", f"{metrics_t5['processing_time']:.1f}s"], 
                        textposition='auto',
                    ))
                    fig_time.update_layout(
                        title="Processing Time", yaxis_title="Seconds", height=300, 
                        template="plotly_white", showlegend=False, font=dict(size=12)
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    fig_words = go.Figure()
                    fig_words.add_trace(go.Bar(
                        x=['BART-large-CNN', 'T5-base'], 
                        y=[metrics_bart['summary_words'], metrics_t5['summary_words']], 
                        marker_color=['#1f77b4', '#7b1fa2'],
                        text=[f"{metrics_bart['summary_words']:,}", f"{metrics_t5['summary_words']:,}"], 
                        textposition='auto',
                    ))
                    fig_words.update_layout(
                        title="Summary Length", yaxis_title="Words", height=300, 
                        template="plotly_white", showlegend=False, font=dict(size=12)
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                
                fig_compression = go.Figure()
                fig_compression.add_trace(go.Bar(
                    x=['BART-large-CNN', 'T5-base'], 
                    y=[metrics_bart['compression_ratio'], metrics_t5['compression_ratio']], 
                    marker_color=['#1f77b4', '#7b1fa2'],
                    text=[f"{metrics_bart['compression_ratio']:.1f}%", f"{metrics_t5['compression_ratio']:.1f}%"], 
                    textposition='auto',
                ))
                fig_compression.update_layout(
                    title="Compression Ratio", yaxis_title="Compression %", height=300, 
                    template="plotly_white", showlegend=False, font=dict(size=12)
                )
                st.plotly_chart(fig_compression, use_container_width=True)
                
                st.markdown("### Performance Analysis")
                
                time_diff = metrics_bart['processing_time'] - metrics_t5['processing_time']
                word_diff = metrics_bart['summary_words'] - metrics_t5['summary_words']
                compression_diff = metrics_bart['compression_ratio'] - metrics_t5['compression_ratio']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if time_diff > 0:
                        st.metric(
                            "T5 Speed Advantage", 
                            f"{abs(time_diff):.1f}s faster", 
                            f"{(abs(time_diff)/metrics_bart['processing_time']*100):.0f}% faster"
                        )
                    else:
                        st.metric(
                            "BART Speed Advantage", 
                            f"{abs(time_diff):.1f}s faster", 
                            f"{(abs(time_diff)/metrics_t5['processing_time']*100):.0f}% faster" if metrics_t5['processing_time'] > 0 else "0%"
                        )
                
                with col2:
                    st.metric(
                        "BART Detail Advantage", 
                        f"+{word_diff:,} words", 
                        f"{(word_diff/metrics_t5['summary_words']*100):.0f}% more content" if metrics_t5['summary_words'] > 0 else "0%"
                    )
                
                with col3:
                    st.metric(
                        "Compression Delta", 
                        f"{abs(compression_diff):.1f}%", 
                        "BART preserves more detail" if compression_diff < 0 else "T5 more aggressive"
                    )

        with tab5:
            st.markdown('<p class="section-header">Question & Answer</p>', unsafe_allow_html=True)
            
            # Build RAG index if not already built
            if "rag_index" not in st.session_state or "rag_chunks" not in st.session_state:
                with st.spinner("Building Q&A index..."):
                    try:
                        index, chunks = build_vector_store(st.session_state["transcript"])
                        st.session_state["rag_index"] = index
                        st.session_state["rag_chunks"] = chunks
                        st.success(f"Q&A system ready ({len(chunks)} chunks indexed)")
                    except Exception as e:
                        st.error(f"Failed to build Q&A index: {str(e)}")
                        st.stop()
            
            st.markdown("**Ask questions about the transcript:**")
            st.caption("The system will find relevant sections and provide answers based on the content.")
            
            # Question input
            question = st.text_input(
                "Your question:",
                placeholder="e.g., What are the main points discussed?",
                key="qa_question"
            )
            
            ask_button = st.button("Ask", type="primary", use_container_width=True)
            
            # Generate answer when button clicked
            if ask_button and question.strip():
                with st.spinner("Finding answer..."):
                    try:
                        answer = generate_answer(
                            question,
                            st.session_state["rag_index"],
                            st.session_state["rag_chunks"]
                        )
                        
                        # Store in session state
                        if "qa_history" not in st.session_state:
                            st.session_state["qa_history"] = []
                        
                        st.session_state["qa_history"].append({
                            "question": question,
                            "answer": answer
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            
            # Display current answer
            if "qa_history" in st.session_state and len(st.session_state["qa_history"]) > 0:
                st.divider()
                
                # Show most recent Q&A
                latest = st.session_state["qa_history"][-1]
                st.markdown(f"**Q: {latest['question']}**")
                st.markdown(latest['answer'])
                
                # Show Q&A history if there are multiple
                if len(st.session_state["qa_history"]) > 1:
                    st.divider()
                    with st.expander(f"View Q&A History ({len(st.session_state['qa_history'])-1} previous)"):
                        for i, qa in enumerate(reversed(st.session_state["qa_history"][:-1]), 1):
                            st.markdown(f"**Q{len(st.session_state['qa_history'])-i}: {qa['question']}**")
                            st.markdown(qa['answer'])
                            st.markdown("---")
                
                # Clear history button
                if st.button("Clear Q&A History", use_container_width=True):
                    st.session_state["qa_history"] = []
                    st.rerun()
            
            # Example questions
            with st.expander("Example Questions"):
                st.markdown("""
                Try asking:
                - What is the main topic discussed?
                - What are the key takeaways?
                - Can you explain [specific concept mentioned]?
                - What examples were provided?
                - What solutions were suggested?
                """)

    else:
        st.info("Process a podcast or video to view results")

st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <small>Powered by OpenAI Whisper, BART-large-CNN, T5-base, yt-dlp, and Google TTS</small>
    </div>
""", unsafe_allow_html=True)