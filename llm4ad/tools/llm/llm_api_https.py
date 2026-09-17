# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import http.client
import json
import time
from typing import Any
from urllib.parse import urlsplit
from ...base import LLM


class HttpsApiFatalError(RuntimeError):
    """A non-retryable API configuration/authentication error."""


class HttpsApiRetryExhaustedError(RuntimeError):
    """Transient API attempts were exhausted for one logical request."""


class HttpsApi(LLM):
    def __init__(
            self,
            host,
            key,
            model,
            timeout=60,
            max_retries=3,
            retry_backoff_seconds=2,
            **kwargs
    ):
        """Https API
        Args:
            host   : HTTPS host or OpenAI-compatible base URL. Examples:
                     api.deepseek.com
                     https://dashscope.aliyuncs.com/compatible-mode/v1
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        # LLM only accepts these two base options.  The remaining kwargs are
        # generation parameters for the HTTP payload.
        do_auto_trim = kwargs.pop('do_auto_trim', True)
        debug_mode = kwargs.pop('debug_mode', False)
        super().__init__(do_auto_trim=do_auto_trim, debug_mode=debug_mode)
        endpoint = str(host).strip()
        if '://' not in endpoint:
            endpoint = f'https://{endpoint}'
        parsed_endpoint = urlsplit(endpoint)
        if parsed_endpoint.scheme != 'https' or not parsed_endpoint.netloc:
            raise ValueError('HttpsApi endpoint must be a valid HTTPS host or base URL.')
        if parsed_endpoint.query or parsed_endpoint.fragment:
            raise ValueError('HttpsApi endpoint must not contain a query or fragment.')

        self._host = parsed_endpoint.netloc
        base_path = parsed_endpoint.path.rstrip('/')
        if not base_path:
            base_path = '/v1'
        if base_path.endswith('/chat/completions'):
            self._request_path = base_path
        else:
            self._request_path = f'{base_path}/chat/completions'
        self._key = key
        self._model = model
        self._timeout = timeout
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_seconds = max(0, float(retry_backoff_seconds))
        self._kwargs = kwargs
        self._cumulative_error = 0
        self._request_count = 0
        self._last_usage = {}
        self._cumulative_usage = {}

    @property
    def last_usage(self) -> dict:
        """Token usage returned by the most recent successful API call."""
        return dict(self._last_usage)

    @property
    def cumulative_usage(self) -> dict:
        """Cumulative numeric usage fields returned by the API."""
        return dict(self._cumulative_usage)

    @property
    def request_count(self) -> int:
        """Number of HTTP requests, including retry attempts."""
        return self._request_count

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """
        Sends a request to the LLM and retrieves the generated response.

        This method supports multiple input formats for backward compatibility:
        1. Explicit 'messages' list via kwargs.
        2. A message list passed directly as the 'prompt'.
        3. Multimodal inputs (text + base64 images).
        4. Simple string prompts.

        Args:
            prompt: The text prompt or a list of message dictionaries.
            **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

        Returns:
            The string content of the LLM's response.
        """
        image64s = kwargs.get('image64s', None)  # List[str]
        messages_input = kwargs.get('messages', None)

        # --- 1. Priority: Explicit messages list ---
        if messages_input is not None:
            if isinstance(messages_input, dict):
                messages = [messages_input]
            else:
                messages = messages_input

        # --- 2. Legacy Support: prompt passed as a pre-constructed list ---
        elif not isinstance(prompt, str):
            messages = prompt

        # --- 3. Construction from String + Optional Images ---
        else:
            text_content = prompt.strip()

            if image64s:
                # Construct multimodal content structure
                content = [{
                    "type": "text",
                    "text": text_content
                }]
                for image in image64s:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        }
                    })
                messages = [{'role': 'user', 'content': content}]

            else:
                # Construct standard text-only message
                messages = [{'role': 'user', 'content': text_content}]

        # Retry transient connection/server errors a bounded number of times.
        for attempt in range(1, self._max_retries + 1):
            conn = None
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)

                # Prepare standard OpenAI-compatible payload
                payload_data = {
                    'max_tokens': self._kwargs.get('max_tokens', 8192),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': messages
                }
                if self._kwargs.get('top_p') is not None:
                    payload_data['top_p'] = self._kwargs['top_p']
                if 'enable_thinking' in self._kwargs:
                    payload_data['enable_thinking'] = bool(
                        self._kwargs['enable_thinking']
                    )
                payload = json.dumps(payload_data)
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                self._request_count += 1
                conn.request('POST', self._request_path, payload, headers)
                res = conn.getresponse()
                response_text = res.read().decode('utf-8')
                status = getattr(res, 'status', 200)
                if not 200 <= status < 300:
                    safe_response = response_text.replace(str(self._key), '<redacted>')
                    message = f'HTTP {status}: {safe_response[:500]}'
                    if 400 <= status < 500 and status not in (408, 429):
                        raise HttpsApiFatalError(message)
                    raise RuntimeError(message)
                data = json.loads(response_text)

                # Extract content from the standard response format
                response = data['choices'][0]['message']['content']
                usage = data.get('usage') or {}
                self._last_usage = usage if isinstance(usage, dict) else {}
                for name, value in self._last_usage.items():
                    if isinstance(value, (int, float)):
                        self._cumulative_usage[name] = self._cumulative_usage.get(name, 0) + value
                # Reset error counter on success
                self._cumulative_error = 0
                return response

            except HttpsApiFatalError:
                self._cumulative_error += 1
                raise
            except Exception as error:
                self._cumulative_error += 1
                if attempt >= self._max_retries:
                    raise HttpsApiRetryExhaustedError(
                        f'{self.__class__.__name__} failed after {attempt} attempt(s). '
                        'Check the API host, model, and service availability.'
                    ) from error
                if self.debug_mode:
                    print(
                        f'{self.__class__.__name__} transient error on attempt '
                        f'{attempt}/{self._max_retries}: {error}'
                    )
                time.sleep(self._retry_backoff_seconds * (2 ** (attempt - 1)))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    # def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
    #     """
    #     Handle message construction:
    #     - If 'messages' is explicitly provided, use it as the payload.
    #     - If 'messages' is None, build it from 'prompt' and 'images':
    #         a) Text only: Wrap prompt in a standard user message format.
    #         b) Multimodal: Combine prompt text and image URLs into a single user message content list.
    #     """
    #     image64s = kwargs.get('image64s', None)  # List[str]
    #     messages_input = kwargs.get('messages', None)   # messages
    #
    #     if messages_input is not None:
    #         if isinstance(messages_input, dict):
    #             messages = [messages_input]  # 单消息包装为列表
    #         else:
    #             messages = messages_input
    #     else:
    #         content = []
    #         content.append({
    #                 "type": "text",
    #                 "text": prompt.strip()
    #             })
    #
    #         if image64s is not None:
    #             for image in image64s:
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{image}",
    #                     }
    #                 })
    #
    #         messages = [{
    #             'role': 'user',
    #             'content': content
    #         }]
    #
    #     while True:
    #         try:
    #             conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
    #             payload = json.dumps({
    #                 'max_tokens': self._kwargs.get('max_tokens', 8192),
    #                 'top_p': self._kwargs.get('top_p', None),
    #                 'temperature': self._kwargs.get('temperature', 1.0),
    #                 'model': self._model,
    #                 'messages': messages
    #             })
    #             headers = {
    #                 'Authorization': f'Bearer {self._key}',
    #                 'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #                 'Content-Type': 'application/json'
    #             }
    #             conn.request('POST', '/v1/chat/completions', payload, headers)
    #             res = conn.getresponse()
    #             data = res.read().decode('utf-8')
    #             data = json.loads(data)
    #             # print(data)
    #             response = data['choices'][0]['message']['content']
    #             if self.debug_mode:
    #                 self._cumulative_error = 0
    #             return response
    #         except Exception as e:
    #             self._cumulative_error += 1
    #             if self.debug_mode:
    #                 if self._cumulative_error == 10:
    #                     raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
    #                                        f'You may check your API host and API key.')
    #             else:
    #                 print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
    #                       f'You may check your API host and API key.')
    #                 time.sleep(2)
    #             continue
