import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
from trainer.agent_tools import TOOLS, TOOL_MAP, MOCK_RESULTS, CHECK_ARGS, parse_tool_calls as _parse_tool_calls, execute_tool as _execute_tool
warnings.filterwarnings('ignore')


def get_tools(names):
    return [TOOL_MAP[n] for n in names if n in TOOL_MAP]


TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"], "expected_tool": "calculate_math"},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"], "expected_tool": "get_current_time"},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"], "expected_tool": "unit_converter"},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"], "expected_tool": "random_number"},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"], "expected_tool": "get_current_weather"},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"], "expected_tool": "get_exchange_rate"},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"], "expected_tool": "translate_text"},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"], "expected_tool": "get_current_weather"},
    {"prompt": "计算 2 的 10 次方", "tools": ["calculate_math", "random_number"], "expected_tool": "calculate_math"},
    {"prompt": "上海现在几度？", "tools": ["get_current_weather", "unit_converter"], "expected_tool": "get_current_weather"},
    {"prompt": "1000 美元能换多少人民币？", "tools": ["get_exchange_rate", "calculate_math"], "expected_tool": "get_exchange_rate"},
    {"prompt": "把 'Hello World' 翻译成中文", "tools": ["translate_text", "text_length"], "expected_tool": "translate_text"},
    {"prompt": "伦敦现在几点？", "tools": ["get_current_time", "get_current_weather"], "expected_tool": "get_current_time"},
    {"prompt": "帮我算 999 除以 3", "tools": ["calculate_math"], "expected_tool": "calculate_math"},
    {"prompt": "纽约天气如何？", "tools": ["get_current_weather", "get_current_time"], "expected_tool": "get_current_weather"},
    {"prompt": "50公斤等于多少磅？", "tools": ["unit_converter", "calculate_math"], "expected_tool": "unit_converter"},
    {"prompt": "帮我搜索一下人工智能的最新进展", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search"},
    {"prompt": "读取 /home/user/readme.txt 文件", "tools": ["read_file", "list_directory"], "expected_tool": "read_file"},
    {"prompt": "列出 /home/user 目录下的文件", "tools": ["list_directory", "read_file"], "expected_tool": "list_directory"},
    {"prompt": "查询销售数据库中价格大于1000的记录", "tools": ["sql_query", "calculate_math"], "expected_tool": "sql_query"},
    {"prompt": "苹果公司（AAPL）的股价是多少？", "tools": ["get_stock_price", "get_exchange_rate"], "expected_tool": "get_stock_price"},
    {"prompt": "从北京到上海怎么走？", "tools": ["get_route", "get_current_weather"], "expected_tool": "get_route"},
    {"prompt": "2025-03-07有什么日程安排？", "tools": ["check_schedule", "create_event"], "expected_tool": "check_schedule"},
    {"prompt": "帮我创建一个会议：项目讨论，2025-03-10 09:00到10:00", "tools": ["create_event", "check_schedule"], "expected_tool": "create_event"},
    {"prompt": "帮我总结一下这段文字：人工智能是计算机科学的一个分支", "tools": ["summarize_text", "text_length"], "expected_tool": "summarize_text"},
    {"prompt": "生成一个10到100的随机数", "tools": ["random_number", "calculate_math"], "expected_tool": "random_number"},
    {"prompt": "统计'你好世界'的字符数", "tools": ["text_length", "translate_text"], "expected_tool": "text_length"},
    {"prompt": "给 zhangsan@example.com 发一封邮件", "tools": ["send_email", "check_schedule"], "expected_tool": "send_email"},
    {"prompt": "告诉我关于北京的信息", "tools": ["get_location_info", "get_current_weather"], "expected_tool": "get_location_info"},
    {"prompt": "帮我运行 print('Hello World')", "tools": ["python_exec", "calculate_math"], "expected_tool": "python_exec"},
    {"prompt": "设置一个60秒的倒计时", "tools": ["countdown_timer", "get_current_time"], "expected_tool": "countdown_timer"},
    {"prompt": "帮我写一个文件到 /home/user/test.txt", "tools": ["write_file", "read_file"], "expected_tool": "write_file"},
    {"prompt": "搜索Python教程", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search"},
    {"prompt": "东京的经纬度是多少？", "tools": ["get_location_info", "get_current_weather"], "expected_tool": "get_location_info"},
    {"prompt": "计算 3.14 乘以 100", "tools": ["calculate_math", "unit_converter"], "expected_tool": "calculate_math"},
    {"prompt": "帮我查一下欧元兑人民币的汇率", "tools": ["get_exchange_rate", "calculate_math"], "expected_tool": "get_exchange_rate"},
    {"prompt": "把'今天天气真好'翻译成英文", "tools": ["translate_text", "text_length"], "expected_tool": "translate_text"},
    {"prompt": "广州天气怎么样？", "tools": ["get_current_weather", "get_route"], "expected_tool": "get_current_weather"},
    {"prompt": "帮我算 1024 除以 8", "tools": ["calculate_math", "random_number"], "expected_tool": "calculate_math"},
    {"prompt": "36.5摄氏度等于多少华氏度？", "tools": ["unit_converter", "calculate_math"], "expected_tool": "unit_converter"},
    {"prompt": "帮我搜索深度学习相关内容", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search"},
    {"prompt": "读取 /home/user/data/config.json", "tools": ["read_file", "list_directory"], "expected_tool": "read_file"},
    {"prompt": "查询员工数据库中的记录", "tools": ["sql_query", "calculate_math"], "expected_tool": "sql_query"},
    {"prompt": "谷歌（GOOGL）的股价是多少？", "tools": ["get_stock_price", "get_exchange_rate"], "expected_tool": "get_stock_price"},
    {"prompt": "从上海到杭州怎么走？", "tools": ["get_route", "get_current_weather"], "expected_tool": "get_route"},
    {"prompt": "帮我看看明天有什么安排", "tools": ["check_schedule", "create_event"], "expected_tool": "check_schedule"},
    {"prompt": "安排一个代码审查会议，下午2点到3点", "tools": ["create_event", "check_schedule"], "expected_tool": "create_event"},
    {"prompt": "帮我总结一段关于Python的文字", "tools": ["summarize_text", "text_length"], "expected_tool": "summarize_text"},
    {"prompt": "生成5个1到10的随机数", "tools": ["random_number", "calculate_math"], "expected_tool": "random_number"},
    {"prompt": "统计 'The quick brown fox' 的字数", "tools": ["text_length", "translate_text"], "expected_tool": "text_length"},
    {"prompt": "给团队发一封周报邮件", "tools": ["send_email", "check_schedule"], "expected_tool": "send_email"},
    {"prompt": "伦敦在哪里？", "tools": ["get_location_info", "get_current_weather"], "expected_tool": "get_location_info"},
    {"prompt": "运行 import math; print(math.pi)", "tools": ["python_exec", "calculate_math"], "expected_tool": "python_exec"},
    {"prompt": "设置一个5分钟的倒计时", "tools": ["countdown_timer", "get_current_time"], "expected_tool": "countdown_timer"},
    {"prompt": "把数据写入 /home/user/output.csv", "tools": ["write_file", "read_file"], "expected_tool": "write_file"},
    {"prompt": "Calculate 500 plus 123", "tools": ["calculate_math", "unit_converter"], "expected_tool": "calculate_math"},
    {"prompt": "What's the weather like in Paris?", "tools": ["get_current_weather", "get_current_time"], "expected_tool": "get_current_weather"},
    {"prompt": "Convert 72 degrees Fahrenheit to Celsius", "tools": ["unit_converter", "calculate_math"], "expected_tool": "unit_converter"},
    {"prompt": "Search for transformer architecture", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search"},
    {"prompt": "Read the file /home/user/notes/meeting.txt", "tools": ["read_file", "summarize_text"], "expected_tool": "read_file"},
    {"prompt": "List files in /home/user/data", "tools": ["list_directory", "read_file"], "expected_tool": "list_directory"},
    {"prompt": "Query the sales database for all records", "tools": ["sql_query", "calculate_math"], "expected_tool": "sql_query"},
    {"prompt": "What's TSLA stock price?", "tools": ["get_stock_price", "get_exchange_rate"], "expected_tool": "get_stock_price"},
    {"prompt": "How to get from New York to Los Angeles?", "tools": ["get_route", "get_current_weather"], "expected_tool": "get_route"},
    {"prompt": "Check my schedule for 2025-03-08", "tools": ["check_schedule", "create_event"], "expected_tool": "check_schedule"},
    {"prompt": "Create a meeting: 技术分享, 2025-03-11 10:00 to 11:30", "tools": ["create_event", "check_schedule"], "expected_tool": "create_event"},
    {"prompt": "Summarize this text about machine learning", "tools": ["summarize_text", "text_length"], "expected_tool": "summarize_text"},
    {"prompt": "Give me a random number between 100 and 999", "tools": ["random_number", "calculate_math"], "expected_tool": "random_number"},
    {"prompt": "Count the characters in '人工智能正在改变世界'", "tools": ["text_length", "translate_text"], "expected_tool": "text_length"},
    {"prompt": "Send an email to team@group.org about the project update", "tools": ["send_email", "check_schedule"], "expected_tool": "send_email"},
    {"prompt": "Where is New York located?", "tools": ["get_location_info", "get_current_weather"], "expected_tool": "get_location_info"},
    {"prompt": "Execute this Python code: x = [1,2,3]; print(sum(x))", "tools": ["python_exec", "calculate_math"], "expected_tool": "python_exec"},
    {"prompt": "Set a timer for 2 minutes", "tools": ["countdown_timer", "get_current_time"], "expected_tool": "countdown_timer"},
    {"prompt": "Write data to /home/user/data/results.json", "tools": ["write_file", "read_file"], "expected_tool": "write_file"},
    {"prompt": "帮我算 88 乘以 12", "tools": ["calculate_math", "unit_converter"], "expected_tool": "calculate_math"},
    {"prompt": "深圳天气如何？", "tools": ["get_current_weather", "get_route"], "expected_tool": "get_current_weather"},
    {"prompt": "100英里等于多少公里？", "tools": ["unit_converter", "calculate_math"], "expected_tool": "unit_converter"},
    {"prompt": "搜索大语言模型相关信息", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search"},
    {"prompt": "查看 /home/user/data/sales.csv", "tools": ["read_file", "sql_query"], "expected_tool": "read_file"},
    {"prompt": "微软（MSFT）股价多少？", "tools": ["get_stock_price", "get_exchange_rate"], "expected_tool": "get_stock_price"},
    {"prompt": "从北京到广州的路线", "tools": ["get_route", "get_current_weather"], "expected_tool": "get_route"},
    {"prompt": "我明天有什么安排？", "tools": ["check_schedule", "create_event"], "expected_tool": "check_schedule"},
    {"prompt": "帮我安排一个需求评审会议", "tools": ["create_event", "check_schedule"], "expected_tool": "create_event"},
    {"prompt": "帮我总结一段关于深度学习的文字", "tools": ["summarize_text", "text_length"], "expected_tool": "summarize_text"},
    {"prompt": "生成一个100到999的随机数", "tools": ["random_number", "calculate_math"], "expected_tool": "random_number"},
    {"prompt": "统计'机器学习很有趣'的字符数", "tools": ["text_length", "translate_text"], "expected_tool": "text_length"},
    {"prompt": "给 lisi@company.com 发送会议通知", "tools": ["send_email", "check_schedule"], "expected_tool": "send_email"},
    {"prompt": "上海在哪个国家？经纬度是多少？", "tools": ["get_location_info", "get_current_weather"], "expected_tool": "get_location_info"},
    {"prompt": "运行 print(2**10)", "tools": ["python_exec", "calculate_math"], "expected_tool": "python_exec"},
    {"prompt": "设置一个10分钟的倒计时", "tools": ["countdown_timer", "get_current_time"], "expected_tool": "countdown_timer"},
    {"prompt": "把结果写入 /home/user/output.txt", "tools": ["write_file", "read_file"], "expected_tool": "write_file"},
    {"prompt": "Calculate sqrt(144)", "tools": ["calculate_math", "random_number"], "expected_tool": "calculate_math"},
    {"prompt": "What time is it in Tokyo?", "tools": ["get_current_time", "get_current_weather"], "expected_tool": "get_current_time"},
    {"prompt": "Convert 25 kg to pounds", "tools": ["unit_converter", "calculate_math"], "expected_tool": "unit_converter"},
    {"prompt": "Search for RLHF training methods", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search"},
    {"prompt": "Read the configuration file", "tools": ["read_file", "list_directory"], "expected_tool": "read_file"},
    {"prompt": "Query the employees database", "tools": ["sql_query", "calculate_math"], "expected_tool": "sql_query"},
    {"prompt": "What's Amazon's stock price?", "tools": ["get_stock_price", "get_exchange_rate"], "expected_tool": "get_stock_price"},
    {"prompt": "Route from Chengdu to Chongqing", "tools": ["get_route", "get_current_weather"], "expected_tool": "get_route"},
    {"prompt": "Any meetings on 2025-03-09?", "tools": ["check_schedule", "create_event"], "expected_tool": "check_schedule"},
    {"prompt": "Schedule a weekly meeting", "tools": ["create_event", "check_schedule"], "expected_tool": "create_event"},
    {"prompt": "Summarize this article about AI", "tools": ["summarize_text", "text_length"], "expected_tool": "summarize_text"},
    {"prompt": "Roll a dice (1-6)", "tools": ["random_number", "calculate_math"], "expected_tool": "random_number"},
    {"prompt": "How many words in 'I love programming'?", "tools": ["text_length", "translate_text"], "expected_tool": "text_length"},
    {"prompt": "Send a code review request email", "tools": ["send_email", "check_schedule"], "expected_tool": "send_email"},
    {"prompt": "Tell me about Sydney", "tools": ["get_location_info", "get_current_weather"], "expected_tool": "get_location_info"},
    {"prompt": "Execute: print(sum(range(1,101)))", "tools": ["python_exec", "calculate_math"], "expected_tool": "python_exec"},
    {"prompt": "Set a 30 second timer", "tools": ["countdown_timer", "get_current_time"], "expected_tool": "countdown_timer"},
    {"prompt": "Save results to /home/user/data/output.json", "tools": ["write_file", "read_file"], "expected_tool": "write_file"},
    {"prompt": "帮我查一下北京和上海的天气，哪个更热？", "tools": ["get_current_weather", "calculate_math"], "expected_tool": "get_current_weather", "multi_turn": True},
    {"prompt": "我有500美元，换成人民币是多少？", "tools": ["get_exchange_rate", "calculate_math"], "expected_tool": "get_exchange_rate", "multi_turn": True},
    {"prompt": "查一下北京天气，然后告诉我从北京到上海怎么走", "tools": ["get_current_weather", "get_route"], "expected_tool": "get_current_weather", "multi_turn": True},
    {"prompt": "搜索人工智能，然后总结搜索结果", "tools": ["web_search", "summarize_text"], "expected_tool": "web_search", "multi_turn": True},
    {"prompt": "查一下AAPL股价，然后换算成人民币", "tools": ["get_stock_price", "get_exchange_rate"], "expected_tool": "get_stock_price", "multi_turn": True},
    {"prompt": "现在几点了？顺便查一下北京和东京的天气", "tools": ["get_current_time", "get_current_weather"], "expected_tool": "get_current_time", "multi_turn": True},
    {"prompt": "读取 /home/user/readme.txt 然后总结内容", "tools": ["read_file", "summarize_text"], "expected_tool": "read_file", "multi_turn": True},
    {"prompt": "生成一个随机数然后计算它的平方", "tools": ["random_number", "calculate_math"], "expected_tool": "random_number", "multi_turn": True},
    {"prompt": "把'你好世界'翻译成英文，然后统计翻译后的字符数", "tools": ["translate_text", "text_length"], "expected_tool": "translate_text", "multi_turn": True},
    {"prompt": "看看今天的日程，然后帮我加一个会议", "tools": ["check_schedule", "create_event"], "expected_tool": "check_schedule", "multi_turn": True},
]


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_call_from_text(content):
    calls = _parse_tool_calls(content)
    if not calls:
        pattern = r'【\s*(\{.*?\})\s*】'
        matches = re.findall(pattern, content, re.DOTALL)
        for m in matches:
            try:
                calls.append(json.loads(m.strip()))
            except Exception:
                pass
    if not calls:
        return None
    tool_calls = []
    for i, data in enumerate(calls):
        tool_calls.append({
            "id": f"call_{i}",
            "name": data.get("name", ""),
            "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)
        })
    return tool_calls if tool_calls else None


def execute_tool_eval(name, args_str=None, args_dict=None):
    try:
        if args_dict:
            args = args_dict
        elif args_str:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        else:
            args = {}
    except Exception:
        args = {}
    result = _execute_tool(name, args)
    if result is None:
        return {"error": f"工具执行失败: {name}"}
    return result


def generate(model, tokenizer, messages, tools, args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    max_turns = 5
    turn = 0
    while turn < max_turns:
        turn += 1
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_call_from_text(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        if args.backend == 'api' and tool_calls:
            tool_calls = [{
                "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
                "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
                "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
            } for tc in tool_calls]
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = execute_tool_eval(name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})
    return messages


def run_benchmark(args, model=None, tokenizer=None, client=None):
    results = {"total": 0, "correct_tool": 0, "tool_called": 0, "args_valid": 0, "multi_turn_total": 0, "multi_turn_success": 0}
    for i, case in enumerate(TEST_CASES):
        results["total"] += 1
        prompt = case["prompt"]
        tool_names = case["tools"]
        expected_tool = case.get("expected_tool", "")
        is_multi = case.get("multi_turn", False)
        if is_multi:
            results["multi_turn_total"] += 1
        tools = get_tools(tool_names)
        print(f'\n{"="*60}')
        print(f'[Test {i+1}/{len(TEST_CASES)}] {prompt}')
        print(f'Expected tool: {expected_tool} | Available: {tool_names}')
        print(f'{"="*60}')
        messages = run_case(prompt, tools, args, model=model, tokenizer=tokenizer, client=client)
        tool_found = False
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                parsed = parse_tool_call_from_text(content)
                if parsed:
                    results["tool_called"] += 1
                    tool_found = True
                    for tc in parsed:
                        if tc["name"] == expected_tool:
                            results["correct_tool"] += 1
                        try:
                            args_dict = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                            if CHECK_ARGS.get(tc["name"], lambda a: True)(args_dict):
                                results["args_valid"] += 1
                        except Exception:
                            pass
                    break
        if is_multi and tool_found:
            tool_call_count = sum(1 for msg in messages if msg.get("role") == "assistant" and parse_tool_call_from_text(msg.get("content", "")))
            if tool_call_count >= 2:
                results["multi_turn_success"] += 1
    print(f'\n{"="*60}')
    print(f'Benchmark Results ({results["total"]} cases)')
    print(f'{"="*60}')
    print(f'Tool Call Rate:    {results["tool_called"]}/{results["total"]} = {results["tool_called"]/max(results["total"],1)*100:.1f}%')
    print(f'Correct Tool Rate: {results["correct_tool"]}/{results["total"]} = {results["correct_tool"]/max(results["total"],1)*100:.1f}%')
    print(f'Valid Args Rate:   {results["args_valid"]}/{results["total"]} = {results["args_valid"]/max(results["total"],1)*100:.1f}%')
    if results["multi_turn_total"] > 0:
        print(f'Multi-turn Rate:   {results["multi_turn_success"]}/{results["multi_turn_total"]} = {results["multi_turn_success"]/max(results["multi_turn_total"],1)*100:.1f}%')
    # 保存结果到 JSON
    results["tool_call_rate"] = results["tool_called"] / max(results["total"], 1)
    results["correct_tool_rate"] = results["correct_tool"] / max(results["total"], 1)
    results["valid_args_rate"] = results["args_valid"] / max(results["total"], 1)
    if results["multi_turn_total"] > 0:
        results["multi_turn_rate"] = results["multi_turn_success"] / max(results["multi_turn_total"], 1)
    os.makedirs("../eval_results", exist_ok=True)
    result_file = f"../eval_results/{getattr(args, '_result_prefix', args.weight)}_{args.hidden_size}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Results saved to {result_file}')
    return results


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出")
    parser.add_argument('--benchmark', default=0, type=int, choices=[0, 1], help="运行完整基准测试（0=交互模式，1=自动评测）")
    args = parser.parse_args()

    model = tokenizer = client = None
    if args.backend == 'local':
        model, tokenizer = init_model(args)
    else:
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    if args.benchmark:
        run_benchmark(args, model=model, tokenizer=tokenizer, client=client)
    else:
        input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
        if input_mode == 0:
            cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES]
        else:
            def get_manual_input():
                prompt = input('💬: ')
                return {"prompt": prompt, "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]} if prompt else None
            cases = iter(lambda: get_manual_input(), None)
        for case in cases:
            if not case or not case.get("prompt"):
                break
            setup_seed(random.randint(0, 31415926))
            if input_mode == 0:
                print(f'📦 可用工具: {case["tool_names"]}\n')
                print(f'💬: {case["prompt"]}')
            run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
            print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
