import boto3
import streamlit as st
from pathlib import Path
from typing import Optional

# Configuration embedded in the script
configs = {
    "page_title": "Bedrock KnowledgeBases with Multimodal LLMs",
    "start_message": "I am a GenAI powered chatbot, how can I help you?",
    "bedrock_region": "eu-central-1",
    "claude_model_params": {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_k": 100
    },
    "kb_configs": {"vectorSearchConfiguration": {"numberOfResults": 5}},
    "multimodal_llms": {
        "Anthropic Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Anthropic Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "Anthropic Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0"
    }
}

# BedrockHandler class
class BedrockHandler:
    def __init__(self, client, model_id: str, params: dict):
        self.params = params
        self.model_id = model_id
        self.client = client

    @staticmethod
    def assistant_message(message: str) -> dict:
        return {"role": "assistant", "content": [{"text": message}]}

    @staticmethod
    def user_message(message: str, context: Optional[str] = None, uploaded_files: Optional[list] = None) -> dict:
        context_message = f"You are a helpful assistant, answer the following question based on the provided context: \n\n {context} \n\n " if context else ""
        new_message = {"role": "user", "content": [{"text": f"{context_message} question: {message}"}]}
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                extension = Path(uploaded_file.name).suffix[1:]
                if extension in ["png", "jpeg", "gif", "webp"]:
                    new_message["content"].append({"image": {"format": extension, "source": {"bytes": bytes_data}}})
                elif extension in ["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]:
                    new_message["content"].append({"document": {"format": extension, "name": "random-doc-name", "source": {"bytes": bytes_data}}})
        
        return new_message

    def invoke_model(self, messages: list) -> dict:
        return self.client.converse(modelId=self.model_id, messages=messages, inferenceConfig={"temperature": 0.0}, additionalModelRequestFields={"top_k": 100})

    def invoke_model_with_stream(self, messages: list) -> dict:
        return self.client.converse_stream(modelId=self.model_id, messages=messages, inferenceConfig={"temperature": 0.0}, additionalModelRequestFields={"top_k": 100})


# KBHandler class
class KBHandler:
    def __init__(self, client, kb_params: dict, kb_id: Optional[str] = None):
        self.client = client
        self.kb_id = kb_id
        self.params = kb_params

    def get_relevant_docs(self, prompt: str) -> list[dict]:
        return self.client.retrieve(retrievalQuery={"text": prompt}, knowledgeBaseId=self.kb_id, retrievalConfiguration=self.params)["retrievalResults"] if self.kb_id else []

    @staticmethod
    def parse_kb_output_to_string(docs: list[dict]) -> str:
        return "\n\n".join(f"Document {i + 1}: {doc['content']['text']}" for i, doc in enumerate(docs))

    @staticmethod
    def parse_kb_output_to_reference(docs: list[dict]) -> dict:
        return {f"Document {i + 1}": {"text": doc["content"]["text"], "metadata": doc["location"], "score": doc["score"]} for i, doc in enumerate(docs)}


# Utility functions
def clear_screen() -> None:
    st.session_state.messages = [
        {"role": "assistant", "content": configs["start_message"]}
    ]
    bedrock_handler.messages = []


def get_all_kbs(all_kb: dict) -> dict[str, str]:
    result = {}
    for kb in all_kb["knowledgeBaseSummaries"]:
        result[kb["name"]] = kb["knowledgeBaseId"]
    return result

# Main function
if __name__ == "__main__":
    # Set up the page configuration
    st.set_page_config(page_title=configs["page_title"])

    # Initialize Boto3 clients for Bedrock agents and runtime
    bedrock_agents_client = boto3.client(
        service_name="bedrock-agent", region_name=configs["bedrock_region"]
    )
    bedrock_agent_runtime_client = boto3.client(
        "bedrock-agent-runtime", region_name=configs["bedrock_region"]
    )
    all_kbs = get_all_kbs(bedrock_agents_client.list_knowledge_bases(maxResults=10))

    # Sidebar UI
    with st.sidebar:
        st.title(configs["page_title"])
        streaming_on = st.toggle("Streaming", value=True)
        uploaded_files = st.file_uploader(
            "Choose one or more images", accept_multiple_files=True
        )
        selected_bedrock_model = st.selectbox(
            "Choose Bedrock model", configs["multimodal_llms"].keys(), index=1
        )
        knoweldge_base_selection = st.selectbox(
            "Choose a Knowledge base", ["None"] + list(all_kbs.keys()), index=0
        )
        st.button("New Chat", on_click=clear_screen, type="primary")

    # Initialize Bedrock handler and knowledge base retriever
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name=configs["bedrock_region"]
    )
    model_id = configs["multimodal_llms"][selected_bedrock_model]

    bedrock_handler = BedrockHandler(
        bedrock_runtime, model_id, configs["claude_model_params"]
    )

    selected_kb = (
        all_kbs[knoweldge_base_selection]
        if knoweldge_base_selection != "None"
        else None
    )
    retriever = KBHandler(
        bedrock_agent_runtime_client, configs["kb_configs"], kb_id=selected_kb
    )

    # Initialize session state for messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": configs["start_message"]}
        ]

    if "bedrock_messages" not in st.session_state.keys():
        st.session_state.bedrock_messages = []

    # Display the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input and model interaction
    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        docs = retriever.get_relevant_docs(prompt)
        context = retriever.parse_kb_output_to_string(docs)
        st.session_state.bedrock_messages.append(
            bedrock_handler.user_message(prompt, context, uploaded_files=uploaded_files)
        )

        full_response = ""
        if streaming_on:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                stream = bedrock_handler.invoke_model_with_stream(
                    st.session_state.bedrock_messages
                ).get("stream")
                if stream:
                    for event in stream:
                        if "contentBlockDelta" in event:
                            full_response += event["contentBlockDelta"]["delta"]["text"]
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
                    with st.expander("Show source details >"):
                        st.write(retriever.parse_kb_output_to_reference(docs))
        else:
            with st.chat_message("assistant"):
                response = bedrock_handler.invoke_model(
                    st.session_state.bedrock_messages
                )
                full_response = response["output"]["message"]["content"][0]["text"]
                st.write(full_response)
                with st.expander("Show source details >"):
                    st.write(retriever.parse_kb_output_to_reference(docs))

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.bedrock_messages.append(
            bedrock_handler.assistant_message(full_response)
        )
