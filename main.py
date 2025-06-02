import argparse
from pathlib import Path
from src.gptwrapper import GPTWrapper

def main(args):
    gpt = GPTWrapper(model_name="gemini-2.5-flash-preview-04-17")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    main(args)