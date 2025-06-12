# academic-assistant
# 🎙️ Voice-Driven Academic Assistant

Voice-Driven Academic Assistant is a tool designed to support researchers in drafting scientific articles using spoken input. The system guides users through each IMRAD section—Introduction, Methods, Results, Discussion—collects speech responses, generates structured text via an LLM, and integrates citations automatically.

---

## 🚀 Highlights

- **Full IMRAD Guidance**  
  The assistant conducts a guided interview to scaffold your draft, ensuring completeness and logical structure.

- **Natural Voice Input**  
  Powered by on-device speech recognition, it captures user ideas seamlessly with minimal latency.

- **Instant Draft Generation**  
  Your spoken answers are transformed into coherent academic paragraphs by a scientific-tuned language model.

- **Built-in Citation Support**  
  Captures keywords from speech, searches scholarly databases, and inserts formatted references automatically.

- **Privacy & Control**  
  All transcription and draft generation can be performed locally or on your private instance.

---

## 🏗️ System Architecture

| Component         | Description |
|------------------|-------------|
| **ASR Engine**     | Uses Vosk for real-time speech-to-text, supporting multiple languages without cloud dependencies. |
| **Question Flow**  | Drives a sequence of prompts for IMRAD; adapts dynamically to prior answers. |
| **Draft Generator**| DeepSeek-powered LLM composes academic text from your responses. |
| **Citation Module**| Listens for keywords, fetches references, and formats in-text citations automatically. |
| **Web Interface**  | Built with Streamlit: displays questions, transcripts, drafts, and allows Word export. |

---

## 🎯 Getting Started

1. **Clone the repo**  
   `git clone https://github.com/adelm134/academic-assistant.git`

2. **Install dependencies**  
   `pip install -r requirements.txt`

3. **Configure environment**  
   Copy `.env.template` to `.env` and fill in necessary settings (e.g., API keys).

4. **Run the assistant**  
   Launch the interface with:  
   `streamlit run app.py`

5. **Begin drafting**  
   - Pick your language (e.g., English, Russian)  
   - Speak answers to guided questions  
   - Review auto-generated text and export your Word draft  

---

## 📊 Evaluation & Feedback

In pilot testing, novice researchers completed a full first draft in under ten minutes—approximately half the time compared to keyboard typing. Users praised the structured workflow, natural voice interaction, and immediate citation support. Transcript errors occurred only with strong accents or background noise.

---

## 🤝 Ethics & Academic Integrity

This project is meant to assist writing—not replace the human author. All users remain responsible for verifying content and citations. We encourage transparent disclosure of AI assistance in the final manuscript. Built-in live citation lookups help reduce factual errors and uphold scholarly standards.

---

## 🛠️ Roadmap

- Strengthen speech recognition for accents and domain terms  
- Implement dynamic, context-aware
