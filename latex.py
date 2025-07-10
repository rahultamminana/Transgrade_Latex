#!/usr/bin/env python
import sys
import os
import warnings
import logging
import requests
import json
import base64
import re
import subprocess
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Configuration
OPENAI_API_KEY = "" # Set your OpenAI API key here
DJANGO_API_BASE_URL = "https://giant-oranges-brush.loca.lt"
DJANGO_IMAGES_API = "https://light-planes-see.loca.lt"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ---
# ### 🖼️ Image Fetching Functions
# ---
def fetch_images_from_db(script_id: str) -> list[str]:
    """Fetch images from Django API with proper error handling."""
    url = f"{DJANGO_IMAGES_API}/script-images/"
    try:
        response = requests.get(url, params={"script_id": script_id}, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            logger.warning(f"No images found for script_id: {script_id}")
            return []
        
        # Sort by page_number and extract image_data
        images = sorted(data, key=lambda x: x.get("page_number", 0))
        image_data_list = []
        
        for img in images:
            image_data = img.get("image_data")
            if image_data:
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',', 1)[1]
                image_data_list.append(image_data)
            else:
                logger.warning(f"No image_data found for page {img.get('page_number', 'unknown')}")
        
        logger.info(f"Successfully fetched {len(image_data_list)} images for script_id: {script_id}")
        return image_data_list
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch images from API: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching images: {e}")
        raise

# ---
# ### 🧠 LaTeX Generation Functions
# ---
def analyze_image_latex(base64_image: str, image_number: int) -> str:
    """Analyze image and convert to LaTeX using OpenAI API."""
    system_prompt = r"""
You are an OCR-to-LaTeX transcription assistant specialized in academic content.

You will be shown images of handwritten academic material, typically from math, physics, or chemistry answer scripts.

Your job is to transcribe the content exactly as written into LaTeX, following these strict rules:

1. DO NOT interpret, guess, or add any content not visible in the image.
2. DO NOT fix, rephrase, correct, or explain anything.
3. Use proper LaTeX math syntax for mathematical expressions (use $$ for display math, $ for inline math).
4. Use \ce{} for chemical formulas and reactions.
5. For diagrams, graphs, or figures, insert: % DIAGRAM: [brief description of what you see]
6. Properly escape LaTeX special characters (\, {, }, %, &, #, ^, _, ~, $).
7. Preserve the layout and order exactly as shown.
8. If text is illegible or unclear, use \text{[illegible]} or \text{[unclear]}.
9. For tables, use proper LaTeX table syntax with tabular environment.
10. Maintain proper spacing and line breaks as shown in the original.

Return ONLY the LaTeX code without any explanations, preamble, or document structure.
""".strip()

    user_prompt = f"Convert the content of image {image_number} to LaTeX code. Transcribe exactly what you see without interpretation."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }}
                ]}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content
        if content is None:
            logger.warning(f"Empty response from OpenAI for image {image_number}")
            return f"% No content extracted from image {image_number}"
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error analyzing image {image_number}: {e}")
        return f"% Error processing image {image_number}: {str(e)}"

def sanitize_latex(content: str) -> str:
    """Clean and sanitize LaTeX content."""
    if not content or content.strip() == "":
        return "% Empty content"
    
    # Remove any document structure that might have been added
    content = re.sub(r"\\documentclass.*?\\begin\{document\}", "", content, flags=re.DOTALL)
    content = re.sub(r"\\end\{document\}", "", content)
    content = re.sub(r"```latex|```", "", content)  # Remove markdown code blocks
    
    # Handle special characters more carefully
    special_chars = {
        '%': r'\%',
        '&': r'\&', 
        '#': r'\#',
        '~': r'\textasciitilde{}'
    }
    
    for char, escaped in special_chars.items():
        # Only replace if not already escaped
        content = re.sub(f'(?<!\\\\){re.escape(char)}', escaped, content)
    
    # Handle underscore more carefully (don't escape in math mode)
    content = re.sub(r'(?<!\\)_(?![0-9{}])', r'\\_', content)
    
    # Replace common degree symbol issues
    content = content.replace("\\degree", "^{\\circ}")
    content = content.replace("°", "^{\\circ}")
    
    # Handle TikZ pictures and diagrams
    content = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", 
                    r"% DIAGRAM: TikZ picture removed", content, flags=re.DOTALL)
    
    # Clean up common formatting issues
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple blank lines
    content = re.sub(r'[ \t]+\n', '\n', content)  # Trailing whitespace
    
    # Fix common math mode issues
    content = re.sub(r'\$\$\$+', '$$', content)  # Multiple dollar signs
    content = re.sub(r'\$\$\s*\$\$', '', content)  # Empty math blocks
    
    return content.strip()

def validate_latex_syntax(content: str) -> tuple[bool, str]:
    """Basic LaTeX syntax validation."""
    try:
        # Check for balanced braces
        brace_count = content.count('{') - content.count('}')
        if brace_count != 0:
            return False, f"Unbalanced braces: {brace_count} extra opening braces"
        
        # Check for balanced math delimiters
        dollar_count = content.count('$')
        if dollar_count % 2 != 0:
            return False, "Unbalanced math delimiters ($)"
        
        # Check for common problematic patterns
        if '\\\\\\' in content:
            return False, "Too many consecutive backslashes"
        
        return True, "Syntax appears valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def generate_complete_latex_document(latex_sections: list, script_id: str) -> str:
    """Generate a complete LaTeX document with proper packages and formatting."""
    
    preamble = r"""
\documentclass[11pt, a4paper]{article}

% Essential packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amsfonts, amssymb}
\usepackage{graphicx}
\usepackage[version=4]{mhchem}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}

% Page setup
\geometry{
    a4paper,
    margin=1in,
    top=1.2in,
    bottom=1.2in
}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Script ID: """ + script_id + r"""}
\fancyhead[R]{OCR Generated Report}
\fancyfoot[C]{\thepage}

% Title formatting
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}

% Custom commands for common elements
\newcommand{\pagemarker}[1]{\vspace{1em}\noindent\textbf{--- Page #1 ---}\vspace{0.5em}}

\begin{document}

% Title page
\begin{center}
    \vspace*{2cm}
    {\Large\bfseries OCR Transcription Report}\\[0.5cm]
    {\large Script ID: """ + script_id + r"""}\\[0.5cm]
    {\normalsize Generated on: \today}\\[2cm]
\end{center}

\newpage
""".strip()

    # Combine all sections
    document_body = "\n\n".join(latex_sections)
    
    # End document
    end_matter = r"""
\end{document}
"""
    
    return preamble + "\n\n" + document_body + "\n\n" + end_matter

# ---
# ### 💾 Django API Integration Functions
# ---
def get_existing_vlmdesc(script_id: str):
    """Retrieve existing vlmdesc data for a script_id to preserve it."""
    try:
        possible_urls = [
            f"{DJANGO_API_BASE_URL}/compare-text/{script_id}/",
            f"{DJANGO_API_BASE_URL}/compare-text/?script_id={script_id}",
            f"{DJANGO_API_BASE_URL}/compare-text/"
        ]
        
        for url in possible_urls:
            try:
                logger.info(f"Trying to retrieve existing vlmdesc data from: {url}")
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, dict):
                        existing_vlmdesc = data.get('vlmdesc', {})
                        logger.info(f"Retrieved existing vlmdesc from dict: {existing_vlmdesc}")
                        return existing_vlmdesc
                        
                    elif isinstance(data, list):
                        logger.info(f"Response is a list with {len(data)} items")
                        for item in data:
                            if isinstance(item, dict):
                                if (item.get('script_id') == script_id or 
                                    item.get('script_id') == int(script_id) if script_id.isdigit() else False):
                                    existing_vlmdesc = item.get('vlmdesc', {})
                                    logger.info(f"Found matching script_id in list, vlmdesc: {existing_vlmdesc}")
                                    return existing_vlmdesc
                        
                        if data and isinstance(data[0], dict):
                            existing_vlmdesc = data[0].get('vlmdesc', {})
                            logger.info(f"No exact match found, using first item vlmdesc: {existing_vlmdesc}")
                            return existing_vlmdesc
                        
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {url}: {str(e)}")
                continue
        
        logger.warning(f"Could not retrieve existing vlmdesc from any endpoint for script_id: {script_id}")
        return {}
    
    except Exception as e:
        logger.warning(f"Error retrieving existing vlmdesc: {str(e)}")
        return {}

def save_latex_to_django(script_id: str, latex_content: str, complete_document: str):
    """Save the LaTeX content to the Django API final_corrected_text field."""
    try:
        url = f"{DJANGO_API_BASE_URL}/compare-text/"
        logger.info(f"Saving LaTeX data to: {url}")

        # Get existing vlmdesc data to preserve it
        existing_vlmdesc = get_existing_vlmdesc(script_id)
        
        # Prepare the payload
        payload = {
            "script_id": script_id,
            "restructured": {
                "final_text": " ",  # Keep existing structure
            },
            "vlmdesc": {
                "vlm_desc": existing_vlmdesc,  # Preserve existing vlmdesc
            },
            "final_corrected_text": {
                "result": latex_content,  # Save LaTeX content here
                "complete_document": complete_document,  # Optional: save complete document too
                "generation_type": "latex_ocr",
                "timestamp": str(requests.get('http://worldtimeapi.org/api/timezone/Etc/UTC').json().get('datetime', 'unknown'))
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            logger.info(f"Successfully saved LaTeX data for script_id: {script_id}")
            return True, response.json()
        else:
            logger.error(f"Failed to save LaTeX data: {response.status_code} - {response.text}")
            return False, f"API error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error saving LaTeX data: {str(e)}")
        return False, f"API request error: {str(e)}"

# ---
# ### 🚀 Core LaTeX Generation Logic
# ---
def generate_latex_from_script(script_id: str, save_to_django: bool = True) -> dict:
    """Main function to generate LaTeX from script images."""
    result = {
        "success": False,
        "script_id": script_id,
        "message": "",
        "latex_content": "",
        "complete_document": "",
        "pages_processed": 0,
        "errors": []
    }
    
    latex_sections = []
    
    try:
        logger.info(f"📥 Fetching images for script_id: {script_id}")
        base64_images = fetch_images_from_db(script_id)
        
        if not base64_images:
            result["message"] = "No images found for the given script_id"
            return result
        
        logger.info(f"Found {len(base64_images)} images to process")
        
        for i, base64_img in enumerate(base64_images, start=1):
            logger.info(f"🖼️ Processing page {i}/{len(base64_images)}")
            
            try:
                # Analyze image
                raw_latex = analyze_image_latex(base64_img, i)
                
                # Sanitize content
                cleaned_latex = sanitize_latex(raw_latex)
                
                # Validate syntax
                is_valid, validation_msg = validate_latex_syntax(cleaned_latex)
                if not is_valid:
                    logger.warning(f"Page {i} syntax warning: {validation_msg}")
                    cleaned_latex += f"\n% Syntax warning: {validation_msg}"
                    result["errors"].append(f"Page {i}: {validation_msg}")
                
                latex_sections.append(f"% ===== Page {i} =====\n{cleaned_latex}\n")
                result["pages_processed"] += 1
                
            except Exception as e:
                error_msg = f"Failed to process page {i}: {str(e)}"
                logger.error(error_msg)
                latex_sections.append(f"% {error_msg}\n")
                result["errors"].append(error_msg)
        
        # Generate content
        latex_content = "\n\n".join(latex_sections)
        complete_document = generate_complete_latex_document(latex_sections, script_id)
        
        result["latex_content"] = latex_content
        result["complete_document"] = complete_document
        
        # Save to Django if requested
        if save_to_django:
            save_success, save_message = save_latex_to_django(script_id, latex_content, complete_document)
            if save_success:
                result["success"] = True
                result["message"] = f"LaTeX generated and saved successfully for script_id {script_id}"
            else:
                result["message"] = f"LaTeX generated but failed to save: {save_message}"
        else:
            result["success"] = True
            result["message"] = f"LaTeX generated successfully for script_id {script_id}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_latex_from_script: {e}")
        result["message"] = f"Error: {str(e)}"
        result["errors"].append(str(e))
        return result

# ---
# ### 🌐 Flask Application
# ---
def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    app.secret_key = 'latex_api_secret_key'
    CORS(app, origins=['http://localhost:3000'])

    @app.route('/')
    def index():
        """Root endpoint with API information."""
        return jsonify({
            "message": "LaTeX OCR Generation API",
            "version": "1.0.0",
            "endpoints": {
                "generate_latex": "/generate_latex/<script_id>",
                "generate_latex_no_save": "/generate_latex/<script_id>/no_save",
                "health": "/health",
                "test_images": "/test_images/<script_id>"
            },
            "description": "Generate LaTeX code from handwritten script images using OCR"
        })

    @app.route('/generate_latex/<script_id>')
    def generate_latex_route(script_id):
        """Generate LaTeX from images and save to Django API."""
        if not script_id:
            return jsonify({
                "success": False,
                "message": "Script ID is required"
            }), 400

        logger.info(f"🚀 Starting LaTeX generation for script_id: {script_id}")
        
        try:
            result = generate_latex_from_script(script_id, save_to_django=True)
            
            status_code = 200 if result["success"] else 500
            return jsonify(result), status_code
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_latex_route: {e}")
            return jsonify({
                "success": False,
                "script_id": script_id,
                "message": f"Unexpected error: {str(e)}",
                "errors": [str(e)]
            }), 500

    @app.route('/generate_latex/<script_id>/no_save')
    def generate_latex_no_save_route(script_id):
        """Generate LaTeX from images without saving to Django API."""
        if not script_id:
            return jsonify({
                "success": False,
                "message": "Script ID is required"
            }), 400

        logger.info(f"🚀 Starting LaTeX generation (no save) for script_id: {script_id}")
        
        try:
            result = generate_latex_from_script(script_id, save_to_django=False)
            
            status_code = 200 if result["success"] else 500
            return jsonify(result), status_code
            
        except Exception as e:
            logger.error(f"Unexpected error in generate_latex_no_save_route: {e}")
            return jsonify({
                "success": False,
                "script_id": script_id,
                "message": f"Unexpected error: {str(e)}",
                "errors": [str(e)]
            }), 500

    @app.route('/test_images/<script_id>')
    def test_images_route(script_id):
        """Test endpoint to check image retrieval."""
        try:
            images = fetch_images_from_db(script_id)
            
            return jsonify({
                "script_id": script_id,
                "images_found": len(images),
                "image_sizes": [len(img) for img in images[:5]],  # First 5 image sizes
                "first_image_preview": images[0][:100] + "..." if images else None,
                "status": "success" if images else "no_images_found"
            })
            
        except Exception as e:
            return jsonify({
                "script_id": script_id,
                "error": str(e),
                "status": "error"
            }), 500

    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        try:
            # Test Django API connectivity
            django_response = requests.get(f"{DJANGO_API_BASE_URL}/", timeout=5)
            django_status = "connected" if django_response.status_code == 200 else "error"
            
            # Test Images API connectivity
            images_response = requests.get(f"{DJANGO_IMAGES_API}/", timeout=5)
            images_status = "connected" if images_response.status_code == 200 else "error"
            
            # Test OpenAI API
            try:
                openai_test = client.models.list()
                openai_status = "connected"
            except Exception:
                openai_status = "error"
            
            overall_status = "healthy" if all([
                django_status == "connected",
                images_status == "connected", 
                openai_status == "connected"
            ]) else "unhealthy"
            
            return jsonify({
                "status": overall_status,
                "services": {
                    "django_api": {
                        "status": django_status,
                        "url": DJANGO_API_BASE_URL
                    },
                    "images_api": {
                        "status": images_status,
                        "url": DJANGO_IMAGES_API
                    },
                    "openai_api": {
                        "status": openai_status
                    }
                }
            })
            
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 500

    return app

# ---
# ### 🧭 Main Entry Point
# ---
def run():
    """Start the Flask application."""
    app = create_app()
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"🚀 Starting LaTeX OCR API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "run":
        run()
    else:
        print("Usage: python latex_api.py run")
        print("\nAvailable endpoints:")
        print("  GET /generate_latex/<script_id> - Generate LaTeX and save to Django")
        print("  GET /generate_latex/<script_id>/no_save - Generate LaTeX without saving")
        print("  GET /test_images/<script_id> - Test image retrieval")
        print("  GET /health - Health check")