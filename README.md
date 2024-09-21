# HLNP-Translate

Helsinki NLP model AI translation tool - translate JSON localizations files automatically.

## About 

A small python tool using Helsinki NLP models to translate the JSON based localizations from one language to another.
Runs locally on your machine, using CPU or CUDA, depending on your machine.

## How to use

Using a python `venv` is recommended.

```
# Install deps Deps 
pip3 install -r requirements.txt
```

```
python3 translate.py [input_json] [output_json] [source_language] [target_language] [target_ext_optional]

# Example English to French
python3 translate.py en.json fr.json en fr

# Example English to Spanish
python3 translate.py en.json es.json en es
```

Translating models with extension target is also supported, NLP models for languages like Baltic or Slavic languages require it.

```
# Slavic languages are SLA code with desired extension code 

# Example English to Slavic Bosnian (lat)
python3 translate.py en.json ba.json en sla bos_Latn

# Example Enlish to Slavic Serbian (cyr)
python3 translate.py en.json sr.json en sla srp_Cyrl
```

## Example of translation 

Input: 

```
{
    "global": {
        "buttonNext" : "Next"
    }
}
```

Output (Spanish): 

```
{
    "global": {
        "buttonNext" : "próxima"
    }
}
```

## Model docs:

* https://huggingface.co/Helsinki-NLP
* https://huggingface.co/Helsinki-NLP/opus-mt-en-sla