# Phone Driver

A Python-based mobile automation agent that uses vision-language models to understand and interact with Android devices through visual analysis and ADB commands. Supports **GPT-5.2 (Copilot/OpenAI API, default)** and **Qwen3-VL (local)** as interchangeable backends.

<p align="center">
  <img src="Images/PhoneDriver.png" width="600" alt="Phone Driver Demo">
</p>

## Features

- 🤖 **Dual backend**: Qwen3-VL-235B via Intel AI Gateway (default, no local GPU) or Qwen3-VL running locally (GPU)
- 📱 **ADB integration**: Controls Android devices via ADB commands
- 🎯 **Natural language tasks**: Describe what you want in plain English
- 🖥️ **Web UI**: Built-in Gradio interface for easy control
- 📊 **Real-time feedback**: Live screenshots and execution logs

## Requirements

- Python 3.10+
- Android device with USB debugging & Developer Mode enabled
- ADB (Android Debug Bridge) installed
- **OpenAI-compatible backend** (default): Access to `http://gateway.aichina.intel.com/v1` — no local GPU required
- **Qwen3-VL backend**: GPU with sufficient VRAM (tested on 24 GB with Qwen3-VL-8B)

## Installation

### 1. Install ADB

**Linux/Ubuntu:**
```bash
sudo apt update
sudo apt install adb
```

### 2. Clone Repo & Install Python Dependencies

```bash
git clone https://github.com/OminousIndustries/PhoneDriver.git
cd PhoneDriver
```

Create a virtual environment:

```bash
python -m venv phonedriver
source phonedriver/bin/activate
```

Install Python dependencies:

```bash
# Core (required for both backends)
pip install openai pillow gradio requests

# Qwen3-VL backend only (local GPU)
pip install torch torchvision
pip install git+https://github.com/huggingface/transformers
pip install qwen_vl_utils
```

### 3. Configure Your Backend

**Intel AI Gateway / OpenAI-compatible (default — no GPU needed)**

The default config already points to the Intel gateway. Optionally set an API key:
```bash
export OPENAI_API_KEY="your-key-if-required"
```
Or set `api_key` directly in `config.json`.

**Qwen3-VL (local GPU)**

Change `config.json`:
```json
{
  "model_backend": "qwen",
  "model_name": "Qwen/Qwen3-VL-8B-Instruct"
}
```

### 4. Connect Your Device

1. Enable USB debugging on your Android device (Settings → Developer Options)
2. Connect via USB
3. Verify connection:
```bash
adb devices
```
You should see your device listed.

## Configuration

### Model Backend

Set `model_backend` in `config.json` (or in the Settings tab of the Web UI):

| `model_backend` | `model_name` example | Notes |
|---|---|---|
| `openai` *(default)* | `qwen3-vl-235b-a22b-instruct-fp8` | Intel AI Gateway; set `api_base` accordingly |
| `qwen` | `Qwen/Qwen3-VL-8B-Instruct` | Local GPU required |

```json
{
  "model_backend": "openai",
  "model_name": "qwen3-vl-235b-a22b-instruct-fp8",
  "api_key": null,
  "api_base": "http://gateway.aichina.intel.com/v1"
}
```

- `api_key` — leave `null` if the endpoint does not require authentication, or set your key (also reads `OPENAI_API_KEY` env var).
- `api_base` — default is `http://gateway.aichina.intel.com/v1`; change to any other OpenAI-compatible endpoint.
- `use_flash_attention` — Qwen backend only; requires `flash_attn` installed.

### Screen Resolution

The agent can auto-detect your device resolution from the Web UI settings tab, but you can manually configure it in `config.json`.

```json
{
  "screen_width": 1080,
  "screen_height": 2340,
  ...
}
```

To get your device resolution, with the device connected to your computer type the following in the terminal: 
```bash
adb shell wm size
```

## Usage

### Web UI (Recommended)

Launch the Gradio interface:

```bash
python ui.py
```

Navigate to `http://localhost:7860` and enter tasks like:
- "Open Chrome"
- "Search for weather in New York"
- "Open Settings and enable WiFi"

### Command Line

```bash
python phone_agent.py "your task here"
```

Example:
```bash
python phone_agent.py "Open the camera app"
```

## How It Works

1. **Screenshot Capture**: Takes a screenshot of the phone via ADB
2. **Visual Analysis**: Qwen3-VL-235B (or local Qwen3-VL) analyses the screen to understand UI elements
3. **Action Planning**: Determines the best action to take (tap, swipe, type, etc.)
4. **Execution**: Sends ADB commands to perform the action
5. **Repeat**: Continues until task is complete or max cycles reached

## Configuration Options

Key settings in `config.json`:

- `model_backend`: `"openai"` (default) or `"qwen"`
- `model_name`: model identifier — `qwen3-vl-235b-a22b-instruct-fp8` for the Intel gateway (default), or `Qwen/Qwen3-VL-8B-Instruct` for local Qwen
- `api_key`: API key if required by the endpoint (or set `OPENAI_API_KEY` env var) — OpenAI-compatible backend only
- `api_base`: API endpoint URL — default `http://gateway.aichina.intel.com/v1` — OpenAI-compatible backend only
- `use_flash_attention`: Enable Flash Attention 2 for faster inference — Qwen backend only
- `temperature`: Model creativity (0.0–1.0, default: 0.1)
- `max_tokens`: Max response length (default: 512)
- `step_delay`: Wait time between actions in seconds (default: 1.5)
- `max_retries`: Maximum retry attempts (default: 3)
- `enable_visual_debug`: Save annotated screenshots for debugging

## Troubleshooting

**Device not detected:**
- Ensure USB debugging is enabled
- Run `adb devices` to verify connection
- Try `adb kill-server && adb start-server`

**Wrong tap locations:**
- Auto-detect resolution in Settings tab of UI
- Or manually verify with `adb shell wm size`

**API authentication errors (OpenAI-compatible backend):**
- Verify `api_base` is set to `http://gateway.aichina.intel.com/v1` (or your endpoint) in `config.json`
- If a key is required, set `api_key` in `config.json` or export `OPENAI_API_KEY`

**Model loading errors (Qwen backend):**
- Ensure you have sufficient VRAM
- Try the 4B model (`Qwen/Qwen3-VL-4B-Instruct`) for lower memory
- Check that `transformers` is installed from source

**Out of memory (Qwen backend):**
- Use a smaller model or reduce `max_tokens`
- Close other applications using GPU memory

## License

Apache License 2.0 - see LICENSE file for details

## Acknowledgments

- Default backend: [Qwen3-VL-235B-A22B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8) served via [Intel AI Gateway](http://gateway.aichina.intel.com/v1)
- Local backend: [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba Cloud
- Uses [Gradio](https://gradio.app/) for the web interface
