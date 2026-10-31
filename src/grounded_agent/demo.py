import argparse
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Final


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Run the grounded agent fixture demo.")
    parser.add_argument("--chat-port", type=int, default=8502)
    parser.add_argument("--gateway-port", type=int, default=4000)
    arguments: Final = parser.parse_args()
    if not all(0 < port < 65536 for port in (arguments.chat_port, arguments.gateway_port)):
        parser.error("Ports must be between 1 and 65535.")
    if arguments.chat_port == arguments.gateway_port:
        parser.error("The chat and gateway ports must differ.")
    chat_port: Final[int] = arguments.chat_port
    gateway_port: Final[int] = arguments.gateway_port
    key: Final = secrets.token_urlsafe(32)
    environment: Final = {
        **os.environ,
        "GROUNDING_DEMO_MODE": "true",
        "GROUNDING_PROVIDER": "judge",
        "LITELLM_API_BASE": f"http://127.0.0.1:{gateway_port}/v1",
        "GROUNDING_FIXTURE_API_BASE": f"http://127.0.0.1:{chat_port}/fixture/v1",
        "LITELLM_API_KEY": key,
        "LITELLM_MASTER_KEY": key,
    }
    chat: Final = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "grounded_agent.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(chat_port),
        ],
        env=environment,
    )
    try:
        gateway: Final = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                str(Path(__file__).with_name("demo-config.yaml")),
                "--host",
                "127.0.0.1",
                "--port",
                str(gateway_port),
            ],
            env=environment,
        )
        try:
            sys.stderr.write(f"Open http://127.0.0.1:{chat_port} after both servers start. Press Ctrl+C to stop.\n")
            while chat.poll() is None and gateway.poll() is None:
                time.sleep(0.5)
            return chat.returncode or gateway.returncode or 1
        except KeyboardInterrupt:
            return 0
        finally:
            _stop(gateway)
    finally:
        _stop(chat)


if __name__ == "__main__":
    sys.exit(main())
