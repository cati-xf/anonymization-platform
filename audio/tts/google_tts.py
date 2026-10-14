from gtts import gTTS

def text_to_speech(text: str, output_path: str) -> str:
    tts = gTTS(text=text, lang='pt', tld='pt')  # lang='pt', tld='pt'
    tts.save(output_path)
    return output_path

if __name__ == "__main__":
    text_to_speech(
        "o joão silva que mora na rua das flores em lisboa trabalha na empresa texolutions e foi visto sair do hospital ",
        output_path="data/output_google_pt.wav"
    )