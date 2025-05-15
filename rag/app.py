from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import os
from dotenv import load_dotenv
import json
import glob
from openai import OpenAI
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure API Key
API_KEY = os.getenv('API_KEY', 'your-api-key')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-6GD0Yv4e0mRGu0lJg6rSpo5sKYohJRqCRBQpoAP0KaNjAow2')
OPENAI_API_ENDPOINT = os.getenv('OPENAI_API_ENDPOINT', 'https://api.red-pill.ai/v1')

# Configure OpenAI client
oai_client = OpenAI(
    base_url=OPENAI_API_ENDPOINT,
    api_key=OPENAI_API_KEY
)

class RetrievalSetting(BaseModel):
    top_k: int = Field(..., description="Maximum number of retrieved results")
    score_threshold: float = Field(..., description="Score limit of relevance", ge=0, le=1)

class Condition(BaseModel):
    name: List[str]
    comparison_operator: str
    value: Optional[str] = None

class MetadataCondition(BaseModel):
    logical_operator: str = "and"
    conditions: List[Condition]

class RetrievalRequest(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting
    metadata_condition: Optional[MetadataCondition] = None

def verify_api_key(auth_header):
    if not auth_header or not auth_header.startswith('Bearer '):
        return False, {"error_code": 1001, "error_msg": "Invalid Authorization header format. Expected 'Bearer <api-key>' format."}
    
    token = auth_header.split(' ')[1]
    if token != API_KEY:
        return False, {"error_code": 1002, "error_msg": "Authorization failed"}
    
    return True, None

@app.route('/retrieval', methods=['POST'])
def retrieval():
    # Verify API Key
    auth_header = request.headers.get('Authorization')
    is_valid, error = verify_api_key(auth_header)
    if not is_valid:
        return jsonify(error), 403

    try:
        data = request.get_json()
        # Validate request data
        req = RetrievalRequest(**data)
    except Exception as e:
        logging.error(f"Invalid request format: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error_code": 400,
            "error_msg": f"Invalid request format: {str(e)}"
        }), 400

    # Process user input with OpenAI
    try:
        logging.info(f"Start processing user query: {req.query}")
        try:
            completion = oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                     Please extract TAGs from the user's questions about Solana blockchain applications and program development to facilitate subsequent retrieval. Here are some examples:
                    <example>
                    User: what is pump.fun's program id?
                    Assistant: ["pump.fun"]
                    </example>
                    <example>
                    User: pump.fun's fee recipient?
                    Assistant: ["pump.fun"]
                    </example>
                    <example>
                    User: how to use jupiter?
                    Assistant: ["jupiter"]
                    </example>
                    Based on the examples above, please extract key tags from the user's question. Tags should be specific project names or keywords.
                     """},
                    {"role": "user", "content": req.query}
                ]
            )
        except Exception as api_error:
            logging.error(f"OpenAI API call failed: {str(api_error)}\n{traceback.format_exc()}")
            raise Exception(f"OpenAI API call failed: {str(api_error)}")
        
        # Parse AI response
        try:
            logging.info(f"OpenAI response content: {completion}")
            tags = json.loads(completion.choices[0].message.content)
            
            if not isinstance(tags, list) or not tags:
                error_msg = f"Invalid AI response format: {completion.choices[0].message.content}"
                logging.error(error_msg)
                raise ValueError(error_msg)
            
            # lowercase tags
            tags = [tag.lower() for tag in tags]
            
            # Recursively search for files in the data directory
            matching_files = []
            for tag in tags:
                pattern = f"data/**/*{tag}*"
                files = glob.glob(pattern, recursive=True)
                logging.info(f"Search pattern '{pattern}' found files: {files}")
                matching_files.extend(files)
            
            # Remove duplicates
            matching_files = list(set(matching_files))
            
            # Build response
            results = []
            if not matching_files:
                error_msg = f"No matching files found, search tags: {tags}"
                logging.error(error_msg)
                # return empty results
                return jsonify({"records": []}), 200

            for file_path in matching_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        logging.info(f"Reading file content: {file_path}")
                    # Get relative path
                    rel_path = os.path.relpath(file_path, 'data')
                    results.append(f"- {rel_path}\n```\n{content}\n```")
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {str(e)}")
            
            # Build return format that meets API documentation requirements
            records = []
            for file_path in matching_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        logging.info(f"Reading file content: {file_path}")
                    # Get relative path
                    rel_path = os.path.relpath(file_path, 'data')
                    records.append({
                        "content": content,
                        "score": 1.0,  # Set to 1.0 since it's an exact match
                        "title": rel_path,
                        "metadata": {
                            "path": file_path,
                            "tags": tags
                        }
                    })
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {str(e)}")
            
            return jsonify({"records": records})
            
        except json.JSONDecodeError:
            return jsonify({
                "error_code": 500,
                "error_msg": "Failed to parse AI response"
            }), 500
            
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error_code": 500,
            "error_msg": f"Error processing request: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Add error handling
    @app.errorhandler(500)
    def internal_server_error(error):
        error_msg = f"Internal Server Error: {str(error)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return jsonify({
            "error_code": 500,
            "error_msg": "Internal server error",
            "detail": str(error)
        }), 500

    app.run(host='0.0.0.0', port=6000)
