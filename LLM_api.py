import requests
import json
import sys
import aiohttp  # 添加aiohttp库
import asyncio

# 创建一个全局的TCP连接池，增加最大连接数
# 默认的连接池限制是100，我们增加到200
# TCP_CONNECTOR = aiohttp.TCPConnector(limit=200)

def query_deepseek(prompt, temperature=None, top_p=None, max_tokens=None, context=None):
    """
    调用 DeepSeek 模型 API 生成文本.

    Args:
        prompt:  用户输入的提示文本.
        temperature:  控制生成文本的随机性 (0.0 - 1.0, 越高越随机).预测的下一个 token 的概率分布
        top_p:  控制采样策略，影响生成文本的多样性 (0.0 - 1.0).模型在生成下一个 token 时，只从概率最高的 K 个候选 token 中进行选择。
        max_tokens:  限制模型生成文本的最大长度.
        context:  用于多轮对话的上下文 (token ID 列表).

    Returns:
        模型的 JSON 响应文本.

        "model": "deepseek-r1:32b":  模型名称。  

        "created_at": "2025-02-25T14:22:36.088785621Z":  创建时间。  表示模型生成回复的时间戳 (UTC 时间)。

        "response": "\u003cthink\u003e\n\n\u003c/think\u003e\n\n 就是response的意思

        "done": true":  完成状态。  true 表示模型已经完成回复生成。

        "done_reason": "stop":  完成原因。  "stop" 表示模型是正常停止生成回复的 (例如，达到了最大 token 限制，或者模型认为回复已经完整)。  其他可能的 done_reason 值可能包括 "length" (达到最大长度限制), "context_window" (超出上下文窗口长度) 等。

        "context": [151644, 14880, 109432, ... , 1773]:  上下文信息 (Token IDs)。  这是一个数字列表，表示模型在生成回复过程中使用的上下文 token 的 ID。  这个字段主要用于多轮对话，  在单轮对话中，您可以忽略它。  在多轮对话中，您可以将上轮对话的 context 值作为参数传递给下轮对话的 API 请求，以保持对话的上下文连贯性。

        "total_duration": 8473918113:  总耗时。  表示从请求开始到回复完成的总时间，单位是纳秒 (nanoseconds)。 

        "load_duration": 6832611324:  模型加载耗时。  表示加载模型到内存中花费的时间，单位是纳秒。  

        "prompt_eval_count": 10":  Prompt Token 数量。  表示输入提示文本 (Prompt) 被分词器 (Tokenizer) 分成了多少个 token。 

        "prompt_eval_duration": 523000000:  Prompt Eval 耗时。  表示模型评估 (处理) Prompt Token 花费的时间，单位是纳秒。  

        "eval_count": 34":  生成 Token 数量。  表示模型生成的回复文本被分词器分成了多少个 token。  

        "eval_duration": 1116000000:  Eval 耗时。  表示模型生成回复 Token 花费的时间，单位是纳秒。 
    """

    url = "http://124.70.148.254:11434/api/generate"  
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "deepseek-r1:32b",
        "prompt": prompt,
        "stream": False  # 设置为 False 以获取完整回复，设置为 True 以流式接收回复，流不了一点，试试就知道了
    }

    #  添加可选参数
    if temperature is not None: 
        data["temperature"] = temperature
    if top_p is not None:
        data["top_p"] = top_p
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if context is not None:
        data["context"] = context

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.text  # 或者 response.json()，根据 Ollama API 的实际返回格式
    else:
        return f"Error: {response.status_code} - {response.text}"

def query_deepseek_stream(prompt, temperature=None, top_p=None, max_tokens=None, context=None):
    """
    调用 DeepSeek 模型 API 生成文本，使用流式响应模式.

    Args:
        prompt:  用户输入的提示文本.
        temperature:  控制生成文本的随机性 (0.0 - 1.0, 越高越随机).
        top_p:  控制采样策略，影响生成文本的多样性 (0.0 - 1.0).
        max_tokens:  限制模型生成文本的最大长度.
        context:  用于多轮对话的上下文 (token ID 列表).

    Returns:
        生成器对象，每次产生一个流式响应片段.
    """
    url = "http://124.70.148.254:11434/api/generate"  
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "deepseek-r1:32b",
        "prompt": prompt,
        "stream": True  # 设置为 True 以流式接收回复
    }

    # 添加可选参数
    if temperature is not None: 
        data["temperature"] = temperature
    if top_p is not None:
        data["top_p"] = top_p
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if context is not None:
        data["context"] = context

    # 使用 stream=True 参数来获取流式响应
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    
    if response.status_code == 200:
        # 返回一个生成器，逐行处理流式响应
        final_context = None
        for line in response.iter_lines():
            if line:
                # 解析 JSON 响应
                chunk = json.loads(line.decode('utf-8'))
                
                # 保存最后一个响应的 context 以便后续对话使用
                if 'context' in chunk:
                    final_context = chunk['context']
                
                # 返回当前响应片段和完整的响应对象
                yield chunk
        
        # 如果需要，可以在这里返回最终的 context
        if final_context:
            yield {"final_context": final_context}
    else:
        yield {"error": f"Error: {response.status_code} - {response.text}"}

def print_stream_response(stream_generator):
    """
    打印流式响应到控制台，实时显示模型生成的文本.

    Args:
        stream_generator: 流式响应生成器.
    
    Returns:
        最终的上下文信息 (如果有).
    """
    final_context = None
    full_response = ""
    
    print("模型响应: ", end="", flush=True)
    for chunk in stream_generator:
        if "error" in chunk:
            print(f"\n{chunk['error']}")
            return None
        
        if "final_context" in chunk:
            final_context = chunk["final_context"]
            continue
            
        if "response" in chunk:
            # 提取当前片段的文本
            response_text = chunk["response"]
            # 只打印新增的部分
            if response_text.startswith(full_response):
                new_text = response_text[len(full_response):]
                print(new_text, end="", flush=True)
            else:
                print(response_text, end="", flush=True)
            full_response = response_text
    
    print("\n")  # 完成后换行
    return final_context

async def query_deepseek_stream_async(prompt, temperature=None, top_p=None, max_tokens=None, context=None):
    """
    异步调用 DeepSeek 模型 API 生成文本，使用流式响应模式.

    Args:
        prompt:  用户输入的提示文本.
        temperature:  控制生成文本的随机性 (0.0 - 1.0, 越高越随机).
        top_p:  控制采样策略，影响生成文本的多样性 (0.0 - 1.0).
        max_tokens:  限制模型生成文本的最大长度.
        context:  用于多轮对话的上下文 (token ID 列表).

    Returns:
        异步生成器对象，每次产生一个流式响应片段.
    """
    url = "http://124.70.148.254:11434/api/generate"  
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "deepseek-r1:32b",
        "prompt": prompt,
        "stream": True  # 设置为 True 以流式接收回复
    }

    # 添加可选参数
    if temperature is not None: 
        data["temperature"] = temperature
    if top_p is not None:
        data["top_p"] = top_p
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if context is not None:
        data["context"] = context

    # 使用全局连接池进行异步请求
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                # 返回一个异步生成器，逐行处理流式响应
                final_context = None
                async for line in response.content:
                    if line:
                        # 解析 JSON 响应
                        line_str = line.decode('utf-8').strip()
                        if line_str:
                            try:
                                chunk = json.loads(line_str)
                                
                                # 保存最后一个响应的 context 以便后续对话使用
                                if 'context' in chunk:
                                    final_context = chunk['context']
                                
                                # 返回当前响应片段
                                yield chunk
                            except json.JSONDecodeError:
                                yield {"error": f"JSON解析错误: {line_str}"}
                
                # 如果需要，可以在这里返回最终的 context
                if final_context:
                    yield {"final_context": final_context}
            else:
                error_text = await response.text()
                yield {"error": f"Error: {response.status} - {error_text}"}

if __name__ == "__main__":
    prompt_text = "请思考当前的大模型发展历程，然后介绍一下 DeepSeek 大模型"
    
    # 根据需要调整以下参数
    temperature_val = 0.7
    top_p_val = 0.9
    max_tokens_val = 200
    context_val = None  # 如果不需要上下文，可以设置为 None
    
    # 使用流式响应
    print("使用流式响应模式:")
    stream_gen = query_deepseek_stream(
        prompt_text,
        temperature=temperature_val,
        top_p=top_p_val,
        max_tokens=max_tokens_val,
        context=context_val
    )
    
    # 打印流式响应并获取最终上下文
    final_context = print_stream_response(stream_gen)
    
    # 打印最终上下文信息（可选）
    if final_context:
        print("\n最终上下文信息 (用于多轮对话):")
        print(f"context_val = {final_context}")
    
    # 如果需要比较，也可以使用非流式模式
    # print("\n使用非流式响应模式:")
    # result = query_deepseek(
    #     prompt_text,
    #     temperature=temperature_val,
    #     top_p=top_p_val,
    #     max_tokens=max_tokens_val,
    #     context=context_val
    # )
    # print(json.loads(result)["response"])