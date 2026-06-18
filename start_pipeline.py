import subprocess
import time
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineStarter")

def run_pipeline(input_file):
    # 1. Run setup script
    logger.info("Running kaggle_setup.sh...")
    subprocess.run(["bash", "kaggle_setup.sh"], check=True)

    # 2. Start Ollama server
    logger.info("Starting Ollama server...")
    process = subprocess.Popen(
        ["ollama", "serve"], 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )

    # 3. Wait for Ollama to boot
    time.sleep(10)
    
    # 4. Pull model
    logger.info("Pulling qwen2.5:7b...")
    subprocess.run(["ollama", "pull", "qwen2.5:7b"], check=True)
    logger.info("Ollama is ready!")

    # 5. Run main.py
    logger.info(f"Running pipeline with input: {input_file}")
    cmd = ["python", "main.py", "novel", "--input", input_file]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="projects/novel/input/chapter1.txt", help="Path to input file")
    args = parser.parse_args()
    
    run_pipeline(args.input)
