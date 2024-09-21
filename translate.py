import json
import argparse
import torch
from transformers import MarianMTModel, MarianTokenizer

def get_marian_model_name(source_lang, target_lang):
    return f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'

def translate_json(input_file, output_file, source_lang, target_lang, target_ext):
    # Load the MarianMT model and tokenizer
    model_name = get_marian_model_name(source_lang, target_lang)
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except OSError:
        print(f"Error: Model not found for {source_lang} to {target_lang}. Please check language codes.")
        return

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Function to translate a single string
    def translate_text(text):
        # For SLA model, we need to prepend the target language token
        rtext = text
        if target_ext != None:
            rtext = f">>{target_ext}<< {text}"
        
        inputs = tokenizer([rtext], return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        translated = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(f"{source_lang}:{target_lang}:{target_ext}: {text} -> {translated_text}")
        return translated_text

    # Recursive function to translate nested JSON
    def translate_nested(obj):
        if isinstance(obj, dict):
            return {k: translate_nested(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [translate_nested(elem) for elem in obj]
        elif isinstance(obj, str):
            return translate_text(obj)
        else:
            return obj

    # Translate the JSON data
    translated_data = translate_nested(data)

    # Save the translated JSON to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    print(f"Translated JSON saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Translate JSON file values using MarianMT model")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    parser.add_argument("source_lang", help="Source language code (e.g., 'en' for English)")
    parser.add_argument("target_lang", help="Target language code (e.g., 'fr' for French)")
    parser.add_argument("target_ext", nargs="?", default=None, help="Target extension (in case of slavic models, e.g., 'bs_Latn' for Bosnian)")

    # Parse arguments
    args = parser.parse_args()

    # Call the translation function with parsed arguments
    translate_json(args.input_file, args.output_file, args.source_lang, args.target_lang, args.target_ext)

if __name__ == "__main__":
    main()