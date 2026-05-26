from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions, extract_topics
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

def run_pipeline(source : str, language : str = "english")->dict:
  print("starting AI video assistant")

  chunks = process_input(source)

  transcript = transcribe_all(chunks, language=language)
  print(f"raw transcription (first 300 characters ) {transcript[:300]}")

  title = generate_title(transcript)


  summary = summarize(transcript)

  action_item = extract_action_items(transcript)

  decisions = extract_key_decisions(transcript)
  questions = extract_questions(transcript)