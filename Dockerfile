# Use the official Python runtime as the base image
# FROM public.ecr.aws/lambda/python:3.12-arm64
FROM python:3.12-slim

# Set the working directory
WORKDIR ${LAMBDA_TASK_ROOT}

COPY . .

# Create directories for models and NLTK data
RUN mkdir -p /nltk_data /spacy_data 

# Set environment variables for NLTK and spaCy
ENV NLTK_DATA=/nltk_data
ENV SPACY_DATA=/spacy_data

# Install any necessary dependencies
RUN python3 -m pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/nltk_data')"  # Download 'punkt' tokenizer

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "predictions.lambda_handler" ]