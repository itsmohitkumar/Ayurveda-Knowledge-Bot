---
# Bedrock-KnowledgeBase-and-Multimodal Public

This repository showcases a solution that integrates Amazon Bedrock's Knowledge Base with multimodal capabilities. It allows for deploying a highly interactive RAG (Retrieval-Augmented Generation) chatbot, leveraging documents stored in Amazon S3, with support for text, image, and structured data.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup Guide](#setup-guide)
  - [Step 1: Create and Configure S3 Source Bucket](#step-1-create-and-configure-s3-source-bucket)
  - [Step 2: Build Bedrock Knowledge Base](#step-2-build-bedrock-knowledge-base)
    - [Optional: Chunking & Parsing for Enhanced Search](#optional-chunking--parsing-for-enhanced-search)
  - [Step 3: Deploying the Multimodal Chatbot to EC2](#step-3-deploying-the-multimodal-chatbot-to-ec2)
  - [Step 4: Validate the Application](#step-4-validate-the-application)
- [Live Demo](#live-demo)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

**Bedrock-KnowledgeBase-and-Multimodal Public** provides a powerful chatbot framework using Amazon Bedrock’s Knowledge Base, capable of answering user queries by retrieving information from an S3-backed source. This chatbot extends the power of large foundation models through retrieval-augmented generation (RAG), offering a scalable solution for businesses and developers to build intelligent, data-driven chatbots.

The integration supports text and multimodal queries by extracting data from structured documents (PDFs, images, spreadsheets) to deliver rich, contextual responses. This guide will walk you through setting up and deploying the chatbot on AWS.

---

## Features

- **Amazon Bedrock Knowledge Base Integration**: Seamlessly connects with Bedrock to deliver accurate, domain-specific answers.
- **Multimodal Data Handling**: Supports text, images, tables, and other structured formats for retrieval.
- **S3-Backed Knowledge Source**: Uses Amazon S3 as a flexible, scalable storage solution for documents.
- **EC2 Hosting**: Deploys a real-time chatbot on an EC2 instance for easy scaling and access.
- **OpenSearch for Retrieval**: Integrates with OpenSearch for fast, vector-based retrieval.

---

## Architecture

The application architecture is designed to be simple yet powerful, consisting of the following core components:

- **Amazon S3**: Stores the document repository (PDFs, images, etc.).
- **Amazon Bedrock**: Hosts and processes data through a pre-trained Knowledge Base.
- **OpenSearch**: Facilitates efficient document retrieval using embeddings for similarity matching.
- **EC2**: Hosts the frontend and backend components of the chatbot using Streamlit.

![Architecture](https://github.com/user-attachments/assets/arch-diagram.png)

---

## Requirements

To get started, you'll need the following:

- AWS Account with Amazon Bedrock access
- AWS CLI configured
- Python 3.12+ installed
- Git installed
- Amazon EC2 instance (Ubuntu 24.04 LTS)
- S3 Bucket and OpenSearch service enabled

---

## Setup Guide

### Step 1: Create and Configure S3 Source Bucket

1. Log in to the AWS Management Console and create an S3 bucket.
2. Upload all relevant documents (PDFs, Word files, images) that will serve as the knowledge base for the chatbot.

```bash
aws s3 mb s3://your-bucket-name --region us-west-2
aws s3 cp /path/to/documents s3://your-bucket-name/ --recursive
```

### Step 2: Build Bedrock Knowledge Base

#### Request Model Access

1. In Amazon Bedrock Console, request access to the necessary models:
   - Claude 3 Sonnet
   - Titan Embeddings G1
   - Titan Text Embeddings V2

#### Create Knowledge Base

1. Once approved, navigate to the **Knowledge Base** section in Amazon Bedrock.
2. Create a new Knowledge Base linked to your S3 bucket.

3. Configure the document types (PDF, images) and parsing settings to optimize retrieval quality.

### Optional: Chunking & Parsing for Enhanced Search

For better performance, especially with large documents, enable chunking and parsing. This feature breaks down complex documents into smaller, manageable pieces, improving the retrieval speed and relevance.

---

### Step 3: Deploying the Multimodal Chatbot to EC2

#### EC2 Instance Configuration

1. Launch an EC2 instance with Ubuntu 24.04 LTS.
2. Ensure security group rules allow SSH (port 22) and HTTP traffic (port 80).

#### Install Required Dependencies

SSH into the EC2 instance and set up the environment:

```bash
sudo apt update
sudo apt-get install -y git python3-pip python3.12-venv

# Clone this repository
git clone https://github.com/your-username/Bedrock-KnowledgeBase-and-Multimodal.git

# Set up Python environment
python3 -m venv ~/bedrock_env
source ~/bedrock_env/bin/activate

# Install required Python packages
cd Bedrock-KnowledgeBase-and-Multimodal
pip install -r requirements.txt
```

#### Configure Knowledge Base ID

Update the Knowledge Base ID in the `variables.py` file:

```python
KNOWLEDGE_BASE_ID = "your-knowledge-base-id"
```

#### Start the Chatbot Application

1. Create a service for Streamlit to automatically start on boot:

```bash
sudo nano /etc/systemd/system/streamlit.service
```

Paste the following content:

```ini
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=ubuntu
Environment='AWS_DEFAULT_REGION=us-west-2'
WorkingDirectory=/home/ubuntu/Bedrock-KnowledgeBase-and-Multimodal
ExecStart=/bin/bash -c 'source /home/ubuntu/bedrock_env/bin/activate && streamlit run streamlit.py --server.port 8501'
Restart=always

[Install]
WantedBy=multi-user.target
```

2. Start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

Now, visit your EC2 public IP in a web browser to access the chatbot interface.

---

### Step 4: Validate the Application

Ask the chatbot various questions related to the documents uploaded to the S3 bucket to validate the setup. Example queries:

- **"What is Amazon EC2?"**
- **"Summarize the key points of this document."**
- **"Explain the architecture of Bedrock."**

---

## Live Demo

A live demo of the deployed chatbot is available [here](http://your-ec2-public-ip).

---

## Contributing

We welcome contributions! If you'd like to improve the project or report bugs, feel free to open an issue or create a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Notes

- Replace placeholders like `your-bucket-name`, `your-knowledge-base-id`, and `your-ec2-public-ip` with actual values.
- Ensure you have the proper permissions in AWS to access and use Amazon Bedrock services.
- The architecture image URL should be replaced with your actual diagram.
```

--- 