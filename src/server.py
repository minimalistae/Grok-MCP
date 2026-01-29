from pathlib import Path
from typing import List, Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from xai_sdk import Client
from xai_sdk.chat import user, system, image, file
from xai_sdk.tools import web_search as xai_web_search, x_search as xai_x_search, code_execution
from .utils import encode_image_to_base64, encode_video_to_base64, extract_usage, build_params, XAI_API_KEY


mcp = FastMCP(name="Grok MCP Server")

# To Claude: return output urls as links

@mcp.tool()
async def list_models():
    
    client = Client(api_key=XAI_API_KEY)
    models_info = []
    
    models_info.append("## Language Models")
    for model in client.models.list_language_models():
        models_info.append(f"- {model.name} ({model.created.ToDatetime().strftime('%d %B %Y')})")
    
    models_info.append("\n## Image Generation Models")
    for model in client.models.list_image_generation_models():
        models_info.append(f"- {model.name} ({model.created.ToDatetime().strftime('%d %B %Y')})")
    
    client.close()
    return "\n".join(models_info)


@mcp.tool()
async def generate_image(
    prompt: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    n: int = 1,
    image_format: str = "url",
    aspect_ratio: Optional[str] = None,
    model: str = "grok-imagine-image"
):
    
    client = Client(api_key=XAI_API_KEY)
    
    params = {"model": model, "prompt": prompt, "n": n, "image_format": image_format}
    
    if image_path:
        base64_string = encode_image_to_base64(image_path)
        ext = Path(image_path).suffix.lower().replace('.', '')
        params["image_url"] = f"data:image/{ext};base64,{base64_string}"
    elif image_url:
        params["image_url"] = image_url
    
    if aspect_ratio:
        params["aspect_ratio"] = aspect_ratio
    
    images = client.image.sample_batch(**params)
    client.close()

    return {"url": [img.url for img in images], "revised_prompts": [img.prompt for img in images]}


@mcp.tool()
async def generate_video(
    prompt: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    video_path: Optional[str] = None,
    video_url: Optional[str] = None,
    duration: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    model: str = "grok-imagine-video"
):
    
    client = Client(api_key=XAI_API_KEY)
    
    params = {
        "model": model,
        "prompt": prompt
    }
    
    if image_path:
        base64_string = encode_image_to_base64(image_path)
        ext = Path(image_path).suffix.lower().replace('.', '')
        params["image_url"] = f"data:image/{ext};base64,{base64_string}"
    elif image_url:
        params["image_url"] = image_url
    
    if video_path:
        base64_string = encode_video_to_base64(video_path)
        ext = Path(video_path).suffix.lower().replace('.', '')
        params["video_url"] = f"data:video/{ext};base64,{base64_string}"
    elif video_url:
        params["video_url"] = video_url
    
    if duration:
        params["duration"] = duration
    if aspect_ratio:
        params["aspect_ratio"] = aspect_ratio
    if resolution:
        params["resolution"] = resolution

    response = client.video.generate(**params)
    client.close()
    
    return {
        "url": response.url,
        "duration": response.duration if hasattr(response, 'duration') else None
    }

@mcp.tool()
async def chat_with_vision(
    prompt: str,
    image_paths: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    detail: str = "auto",
    model: str = "grok-4"
):

    client = Client(api_key=XAI_API_KEY)
    chat = client.chat.create(model=model, store_messages=False)
    
    user_content = []
    if image_paths:
        for path in image_paths:
            ext = Path(path).suffix.lower().replace('.', '')
            if ext not in ["jpg", "jpeg", "png"]:
                raise ValueError(f"Unsupported image type: {ext}")
            base64_img = encode_image_to_base64(path)
            user_content.append(image(image_url=f"data:image/{ext};base64,{base64_img}", detail=detail))
    
    if image_urls:
        for url in image_urls:
            user_content.append(image(image_url=url, detail=detail))
    
    user_content.append(prompt)
    chat.append(user(*user_content))
    response = chat.sample()
    client.close()
    
    return {
        "content": response.content,
        "usage": {
            "prompt_text_tokens": response.usage.prompt_text_tokens,
            "prompt_image_tokens": response.usage.prompt_image_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else {},
    }


@mcp.tool()
async def chat(
    prompt: str,
    model: str = "grok-4",
    system_prompt: Optional[str] = None,
    store_messages: bool = False
):

    client = Client(api_key=XAI_API_KEY)
    chat = client.chat.create(model=model, store_messages=store_messages)
    if system_prompt:
        chat.append(system(system_prompt))
    chat.append(user(prompt))
    response = chat.sample()
    client.close()

    return {
        "content": response.content,
        "usage": extract_usage(response),
    }


@mcp.tool()
async def chat_with_reasoning(
    prompt: str,
    model: str = "grok-3-mini",
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None
):
    #for seeing reasoning content besides grok-3-mini model use stateful chat

    client = Client(api_key=XAI_API_KEY, timeout=3600)
    
    chat_params = {"model": model}
    if reasoning_effort:
        chat_params["reasoning_effort"] = reasoning_effort
    
    chat = client.chat.create(**chat_params)
    if system_prompt:
        chat.append(system(system_prompt))
    chat.append(user(prompt))
    response = chat.sample()
    client.close()
    
    return {
        "content": response.content,
        "reasoning_content": getattr(response, 'reasoning_content', None),
        "usage": extract_usage(response),
    }


@mcp.tool()
async def web_search(
    prompt: str,
    model: str = "grok-4-1-fast",
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    enable_image_understanding: bool = False,
    include_inline_citations: bool = False,
    max_turns: Optional[int] = None
):

    if allowed_domains and excluded_domains:
        raise ValueError("Cannot specify both allowed_domains and excluded_domains")
    if allowed_domains and len(allowed_domains) > 5:
        raise ValueError("allowed_domains max 5")
    if excluded_domains and len(excluded_domains) > 5:
        raise ValueError("excluded_domains max 5")
    
    client = Client(api_key=XAI_API_KEY)
    
    tool_params = build_params(
        allowed_domains=allowed_domains,
        excluded_domains=excluded_domains,
        enable_image_understanding=enable_image_understanding,
    )
    
    include_options = []
    if include_inline_citations:
        include_options.append("inline_citations")
    
    chat_params = {"model": model, "tools": [xai_web_search(**tool_params)]}
    if include_options:
        chat_params["include"] = include_options
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    chat.append(user(prompt))
    response = chat.sample()
    
    result = {
        "content": response.content,
        "citations": list(response.citations) if response.citations else [],
        "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in response.tool_calls],
        "usage": extract_usage(response),
        "server_side_tool_usage": dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}
    }
    
    if include_inline_citations and response.inline_citations:
        result["inline_citations"] = [
            {"id": c.id, "url": c.web_citation.url if c.HasField("web_citation") else c.x_citation.url}
            for c in response.inline_citations
        ]
    
    client.close()
    return result


@mcp.tool()
async def x_search(
    prompt: str,
    model: str = "grok-4-1-fast",
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
    include_inline_citations: bool = False,
    max_turns: Optional[int] = None
):

    if allowed_x_handles and excluded_x_handles:
        raise ValueError("Cannot specify both allowed_x_handles and excluded_x_handles")
    if allowed_x_handles and len(allowed_x_handles) > 10:
        raise ValueError("allowed_x_handles max 10")
    if excluded_x_handles and len(excluded_x_handles) > 10:
        raise ValueError("excluded_x_handles max 10")
    
    client = Client(api_key=XAI_API_KEY)
    
    tool_params = build_params(
        allowed_x_handles=allowed_x_handles,
        excluded_x_handles=excluded_x_handles,
        from_date=datetime.strptime(from_date, "%d-%m-%Y") if from_date else None,
        to_date=datetime.strptime(to_date, "%d-%m-%Y") if to_date else None,
        enable_image_understanding=enable_image_understanding,
        enable_video_understanding=enable_video_understanding,
    )
    
    include_options = []
    if include_inline_citations:
        include_options.append("inline_citations")
    
    chat_params = {"model": model, "tools": [xai_x_search(**tool_params)]}
    if include_options:
        chat_params["include"] = include_options
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    chat.append(user(prompt))
    response = chat.sample()
    
    result = {
        "content": response.content,
        "citations": list(response.citations) if response.citations else [],
        "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in response.tool_calls],
        "usage": extract_usage(response),
        "server_side_tool_usage": dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}
    }
    
    if include_inline_citations and response.inline_citations:
        result["inline_citations"] = [
            {"id": c.id, "url": c.x_citation.url if c.HasField("x_citation") else c.web_citation.url}
            for c in response.inline_citations
        ]
    
    client.close()
    return result


@mcp.tool()
async def code_executor(
    prompt: str,
    model: str = "grok-4-1-fast",
    max_turns: Optional[int] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    chat_params = {"model": model, "tools": [code_execution()], "include": ["code_execution_call_output"]}
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    chat.append(user(prompt))
    response = chat.sample()
    
    client.close()
    
    result = {
        "content": response.content,
        "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in response.tool_calls],
        "usage": extract_usage(response),
        "server_side_tool_usage": dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}
    }

    if response.tool_outputs:
        result["code_outputs"] = [tool_output.message.content for tool_output in response.tool_outputs]
    
    return result


@mcp.tool()
async def grok_agent(
    prompt: str,
    file_ids: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    use_web_search: bool = False,
    use_x_search: bool = False,
    use_code_execution: bool = False,
    allowed_domains: Optional[List[str]] = None,
    excluded_domains: Optional[List[str]] = None,
    allowed_x_handles: Optional[List[str]] = None,
    excluded_x_handles: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
    include_inline_citations: bool = False,
    model: str = "grok-4-1-fast",
    system_prompt: Optional[str] = None,
    max_turns: Optional[int] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    tools = []
    if use_web_search:
        web_params = build_params(
            allowed_domains=allowed_domains,
            excluded_domains=excluded_domains,
            enable_image_understanding=enable_image_understanding,
        )
        tools.append(xai_web_search(**web_params))
    
    if use_x_search:
        x_params = build_params(
            allowed_x_handles=allowed_x_handles,
            excluded_x_handles=excluded_x_handles,
            from_date=datetime.strptime(from_date, "%d-%m-%Y") if from_date else None,
            to_date=datetime.strptime(to_date, "%d-%m-%Y") if to_date else None,
            enable_image_understanding=enable_image_understanding,
            enable_video_understanding=enable_video_understanding,
        )
        tools.append(xai_x_search(**x_params))
    
    if use_code_execution:
        tools.append(code_execution())
    
    include_options = ["code_execution_call_output"]
    if include_inline_citations:
        include_options.append("inline_citations")
    
    chat_params = {"model": model, "include": include_options}
    if tools:
        chat_params["tools"] = tools
    if max_turns:
        chat_params["max_turns"] = max_turns
    
    chat = client.chat.create(**chat_params)
    
    if system_prompt:
        chat.append(system(system_prompt))
    
    content_items = []
    
    if file_ids:
        for fid in file_ids:
            content_items.append(file(fid))
    
    if image_urls:
        for url in image_urls:
            content_items.append(image(image_url=url))
    
    if image_paths:
        for path in image_paths:
            ext = Path(path).suffix.lower().replace('.', '')
            base64_img = encode_image_to_base64(path)
            content_items.append(image(image_url=f"data:image/{ext};base64,{base64_img}"))
    
    content_items.append(prompt)
    chat.append(user(*content_items))
    response = chat.sample()
    
    client.close()
    
    result = {
        "content": response.content,
        "usage": extract_usage(response),
    }
    
    if response.citations:
        result["citations"] = list(response.citations)
    
    if response.tool_calls:
        result["tool_calls"] = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in response.tool_calls]
    
    if response.server_side_tool_usage:
        result["server_side_tool_usage"] = dict(response.server_side_tool_usage)
    
    if include_inline_citations and response.inline_citations:
        result["inline_citations"] = [
            {"id": c.id, "url": c.web_citation.url if c.HasField("web_citation") else c.x_citation.url}
            for c in response.inline_citations
        ]
    
    return result


@mcp.tool()
async def stateful_chat(
    prompt: str,
    response_id: Optional[str] = None,
    model: str = "grok-4",
    system_prompt: Optional[str] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    chat_params = {"model": model, "store_messages": True}
    if response_id:
        chat_params["previous_response_id"] = response_id
    
    chat = client.chat.create(**chat_params)
    if system_prompt and not response_id:
        chat.append(system(system_prompt))
    chat.append(user(prompt))
    
    response = chat.sample()
    client.close()
    
    return {
        "content": response.content,
        "response_id": response.id,
        "usage": extract_usage(response)
    }


@mcp.tool()
async def retrieve_stateful_response(response_id: str):
    client = Client(api_key=XAI_API_KEY)
    responses = client.chat.get_stored_completion(response_id)
    client.close()
    if not responses:
        return {"error": f"No response found for id {response_id}"}
    response = responses[0] if isinstance(responses, list) else responses
    return {"response_id": response.id, "content": response.content}


@mcp.tool()
async def delete_stateful_response(response_id: str):
    client = Client(api_key=XAI_API_KEY)
    client.chat.delete_stored_completion(response_id)
    client.close()
    return {"response_id": response_id, "deleted": True}


@mcp.tool()
async def upload_file(
    file_path: str,
    filename: Optional[str] = None
):
    client = Client(api_key=XAI_API_KEY)
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found {file_path}")
    
    uploaded = client.files.upload(file_path)
    client.close()
    
    return {
        "file_id": uploaded.id,
        "filename": uploaded.filename,
        "size": uploaded.size,
        "created_at": str(uploaded.created_at),
    }


@mcp.tool()
async def list_files(
    limit: int = 100,
    order: str = "desc",
    sort_by: str = "created_at"
):
    client = Client(api_key=XAI_API_KEY)
    response = client.files.list(limit=limit, order=order, sort_by=sort_by)
    client.close()
    
    return {
        "files": [
            {
                "file_id": f.id,
                "filename": f.filename,
                "size": f.size,
                "created_at": str(f.created_at),
            }
            for f in response.data
        ]
    }


@mcp.tool()
async def get_file(file_id: str):
    client = Client(api_key=XAI_API_KEY)
    file_info = client.files.get(file_id)
    client.close()
    
    return {
        "file_id": file_info.id,
        "filename": file_info.filename,
        "size": file_info.size,
        "created_at": str(file_info.created_at),
    }


@mcp.tool()
async def get_file_content(file_id: str, max_bytes: int = 500000):
    client = Client(api_key=XAI_API_KEY)
    content = client.files.content(file_id)
    client.close()
    
    total_size = len(content)
    truncated = total_size > max_bytes
    
    if truncated:
        content = content[:max_bytes]
    
    return {
        "file_id": file_id,
        "content": content.decode("utf-8", errors="replace"),
        "size": total_size,
        "truncated": truncated,
        "returned_bytes": len(content),
    }


@mcp.tool()
async def delete_file(file_id: str):
    client = Client(api_key=XAI_API_KEY)
    delete_response = client.files.delete(file_id)
    client.close()
    
    return {"file_id": delete_response.id, "deleted": delete_response.deleted}


@mcp.tool()
async def chat_with_files(
    prompt: str,
    file_ids: List[str],
    model: str = "grok-4-1-fast",
    system_prompt: Optional[str] = None
):
    client = Client(api_key=XAI_API_KEY)
    chat = client.chat.create(model=model)
    
    if system_prompt:
        chat.append(system(system_prompt))
    
    file_attachments = [file(fid) for fid in file_ids]
    chat.append(user(prompt, *file_attachments))
    response = chat.sample()
    
    client.close()
    
    return {
        "content": response.content,
        "citations": list(response.citations) if response.citations else [],
        "usage": extract_usage(response),
    }


def main():
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
