# gpt_vl_agent.py
import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import OpenAI
from PIL import Image


# ---------------------------------------------------------------------------
# Tool definition – kept identical to the original Qwen3-VL mobile_use spec
# so that phone_agent.py action parsing is unchanged.
# ---------------------------------------------------------------------------
MOBILE_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "mobile_use",
        "description": (
            "Use a touchscreen to interact with a mobile device, and take screenshots.\n"
            "* This is an interface to a mobile device with touchscreen. You can perform "
            "actions like clicking, typing, swiping, etc.\n"
            "* Some applications may take time to start or process actions, so you may "
            "need to wait and take successive screenshots to see the results of your actions.\n"
            "* The screen's resolution is 999x999.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
            "center of the element. Don't click boxes on their edges unless asked."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["click", "swipe", "type", "wait", "terminate"],
                    "description": (
                        "The action to perform:\n"
                        "* `click`: Click at coordinate (x, y).\n"
                        "* `swipe`: Swipe from (x, y) to (x2, y2).\n"
                        "* `type`: Input text into the active input box.\n"
                        "* `wait`: Wait for the specified number of seconds.\n"
                        "* `terminate`: Finish the task and report status."
                    ),
                },
                "coordinate": {
                    "type": "array",
                    "description": "(x, y) for click/swipe start. Range 0-999.",
                },
                "coordinate2": {
                    "type": "array",
                    "description": "(x, y) swipe end point. Range 0-999.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type. Required for action=type.",
                },
                "time": {
                    "type": "number",
                    "description": "Seconds to wait. Required for action=wait.",
                },
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "Task outcome. Required for action=terminate.",
                },
            },
            "required": ["action"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a mobile GUI automation agent. "
    "Analyse the provided screenshot, reason briefly about the current state, "
    "then call the `mobile_use` tool with the single best next action. "
    "Rules:\n"
    "- Always call mobile_use; never respond with plain text only.\n"
    "- Coordinates are in 999×999 space where (0,0) is top-left.\n"
    "- When the task is done or cannot be completed, use action=terminate.\n"
    "- Be concise in any text you include before the tool call."
)


def _encode_image(image: Image.Image, max_size: int = 1280) -> str:
    """Resize *image* if needed and return a base64-encoded PNG data-URI."""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(d * ratio) for d in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class GPTVLAgent:
    """
    Vision-Language agent using GPT-5.2 (Copilot) for mobile GUI automation.
    Replaces the local Qwen3-VL model with an OpenAI-compatible API call.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        # Accepted for drop-in compatibility but ignored (no local GPU needed)
        device_map: str = "auto",
        dtype=None,
        use_flash_attention: bool = False,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)
        logging.info(f"GPT-VL agent initialized with model: {model_name}")

    # ------------------------------------------------------------------
    # Public interface (same signatures as the original QwenVLAgent)
    # ------------------------------------------------------------------

    def analyze_screenshot(
        self,
        screenshot_path: str,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyse a phone screenshot and determine the next action."""
        try:
            image = Image.open(screenshot_path)
            image_url = _encode_image(image)

            history: List[str] = []
            if context:
                for i, act in enumerate(context.get("previous_actions", [])[-5:], 1):
                    action_type = act.get("action", "unknown")
                    element = act.get("elementName", "")
                    history.append(f"Step {i}: {action_type} {element}".strip())
            history_str = "; ".join(history) if history else "No previous actions"

            user_text = (
                f"User task: {user_request}\n"
                f"Task progress (previous actions): {history_str}"
            )

            messages = self._build_messages(user_text, image_url)
            action = self._generate_action(messages)

            if action:
                logging.info(f"Generated action: {action.get('action', 'unknown')}")
            return action

        except Exception as e:
            logging.error(f"Error analysing screenshot: {e}", exc_info=True)
            return None

    def check_task_completion(
        self,
        screenshot_path: str,
        user_request: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the model whether the task has been completed."""
        try:
            image = Image.open(screenshot_path)
            image_url = _encode_image(image)

            n_actions = len(context.get("previous_actions", []))
            user_text = (
                f"User task: {user_request}\n"
                f"You have performed {n_actions} action(s) so far.\n"
                "Look at the current screen: has the task been completed successfully?\n"
                "Call mobile_use with action=terminate and the appropriate status."
            )

            messages = self._build_messages(user_text, image_url)
            action = self._generate_action(messages)

            if action and action.get("action") == "terminate":
                return {
                    "complete": action.get("status") == "success",
                    "reason": action.get("message", ""),
                    "confidence": 0.9 if action.get("status") == "success" else 0.7,
                }
            return {"complete": False, "reason": "Unable to verify", "confidence": 0.0}

        except Exception as e:
            logging.error(f"Error checking completion: {e}")
            return {"complete": False, "reason": f"Error: {str(e)}", "confidence": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_text: str, image_url: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ],
            },
        ]

    def _generate_action(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call the GPT API and return a parsed action dict."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[MOBILE_USE_TOOL],
                tool_choice={"type": "function", "function": {"name": "mobile_use"}},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            message = response.choices[0].message

            # Log any reasoning text
            if message.content:
                logging.debug(f"Model reasoning: {message.content}")

            if not message.tool_calls:
                logging.error("Model did not return a tool call")
                return None

            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            logging.debug(f"Tool call arguments: {args}")

            return self._parse_action(args, reasoning=message.content or "")

        except Exception as e:
            logging.error(f"Error generating action: {e}", exc_info=True)
            return None

    def _parse_action(
        self, args: Dict[str, Any], reasoning: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Convert raw tool-call arguments to the internal action format."""
        try:
            action_type = args.get("action")
            if not action_type:
                logging.error("No 'action' key in tool-call arguments")
                return None

            action: Dict[str, Any] = {"action": action_type}

            # Coordinates: convert from 999×999 space to normalised 0-1
            if "coordinate" in args:
                coord = args["coordinate"]
                action["coordinates"] = [coord[0] / 999.0, coord[1] / 999.0]

            if "coordinate2" in args:
                coord2 = args["coordinate2"]
                action["coordinate2"] = [coord2[0] / 999.0, coord2[1] / 999.0]

            # Swipe direction helper (for ADB compatibility in phone_agent.py)
            if action_type == "swipe" and "coordinates" in action and "coordinate2" in action:
                start = action["coordinates"]
                end = action["coordinate2"]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                action["direction"] = (
                    ("down" if dy > 0 else "up") if abs(dy) >= abs(dx)
                    else ("right" if dx > 0 else "left")
                )

            # Rename click → tap (internal convention)
            if action_type == "click":
                action["action"] = "tap"
                if "coordinates" not in action:
                    logging.error("click action missing coordinate")
                    return None

            if "text" in args:
                action["text"] = args["text"]
            if "time" in args:
                action["waitTime"] = int(float(args["time"]) * 1000)  # ms
            if "status" in args:
                action["status"] = args["status"]
                action["message"] = f"Task {args['status']}"

            # Attach reasoning from the model's text content
            if reasoning:
                action["reasoning"] = reasoning.strip()

            # Basic validation
            if action["action"] == "tap" and "coordinates" not in action:
                logging.error("tap action missing coordinates")
                return None
            if action["action"] == "type" and "text" not in action:
                logging.error("type action missing text")
                return None

            return action

        except Exception as e:
            logging.error(f"Error parsing action: {e}")
            return None
