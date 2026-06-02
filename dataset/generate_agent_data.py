"""
Agent RL 训练数据生成脚本
基于 agent_tools.py 中的工具定义，自动生成多轮工具调用对话数据

生成数据格式 (agent_rl.jsonl):
{
  "conversations": [
    {"role": "system", "content": "...", "tools": "[...]"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "gt": "期望的最终回答"
}

用法:
  python dataset/generate_agent_data.py --output dataset/agent_rl.jsonl --num_samples 10000
  python dataset/generate_agent_data.py --output dataset/agent_rl.jsonl --num_samples 5000 --multi_turn_ratio 0.6
"""
import json
import os
import sys
import random
import argparse
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.agent_tools import TOOLS, TOOL_MAP, MOCK_RESULTS, CHECK_ARGS, execute_tool

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SINGLE_TURN_TEMPLATES = [
    {"tool": "calculate_math", "templates": [
        ("帮我算一下 {expression}", {"expression": "{expr}"}),
        ("计算 {expression} 的结果", {"expression": "{expr}"}),
        ("请问 {expression} 等于多少", {"expression": "{expr}"}),
        ("What is {expression}?", {"expression": "{expr}"}),
    ], "data": {"expr": ["256*37", "1024/8", "999+1", "100-47", "2**10", "sqrt(144)", "3.14*100", "500+123", "1000-456", "88*12"]}},
    {"tool": "get_current_weather", "templates": [
        ("{location}今天天气怎么样？", {"location": "{city}"}),
        ("查一下{location}的天气", {"location": "{city}"}),
        ("What's the weather in {location}?", {"location": "{city}"}),
        ("告诉我{location}的天气情况", {"location": "{city}"}),
    ], "data": {"city": ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "Tokyo", "New York", "London", "Paris", "Sydney"]}},
    {"tool": "get_current_time", "templates": [
        ("现在几点了？", {"timezone": "Asia/Shanghai"}),
        ("What time is it now?", {"timezone": "America/New_York"}),
        ("伦敦现在几点？", {"timezone": "Europe/London"}),
        ("东京时间是多少？", {"timezone": "Asia/Tokyo"}),
        ("巴黎现在几点？", {"timezone": "Europe/Paris"}),
    ], "data": {}},
    {"tool": "get_exchange_rate", "templates": [
        ("查一下{from_c}兑{to_c}的汇率", {"from_currency": "{from_c}", "to_currency": "{to_c}"}),
        ("1{from_c}等于多少{to_c}？", {"from_currency": "{from_c}", "to_currency": "{to_c}"}),
        ("What's the exchange rate from {from_c} to {to_c}?", {"from_currency": "{from_c}", "to_currency": "{to_c}"}),
    ], "data": {"from_c": ["USD", "EUR", "GBP", "JPY"], "to_c": ["CNY", "EUR", "USD", "GBP"]}},
    {"tool": "translate_text", "templates": [
        ("把'{text}'翻译成{lang}", {"text": "{text}", "target_language": "{lang}"}),
        ("Translate '{text}' to {lang}", {"text": "{text}", "target_language": "{lang}"}),
        ("'{text}'用{lang}怎么说？", {"text": "{text}", "target_language": "{lang}"}),
    ], "data": {"text": ["你好世界", "Good morning", "今天天气真好", "I love programming", "机器学习很有趣", "Happy birthday", "谢谢", "Hello", "再见", "How are you"], "lang": ["english", "chinese", "japanese", "french"]}},
    {"tool": "unit_converter", "templates": [
        ("把{value}{from_u}换算成{to_u}", {"value": "{value}", "from_unit": "{from_u}", "to_unit": "{to_u}"}),
        ("{value}{from_u}是多少{to_u}？", {"value": "{value}", "from_unit": "{from_u}", "to_unit": "{to_u}"}),
        ("Convert {value} {from_u} to {to_u}", {"value": "{value}", "from_unit": "{from_u}", "to_unit": "{to_u}"}),
    ], "data": {"value": [1, 10, 100, 50, 25, 36.5, 0, 72], "from_u": ["km", "miles", "kg", "pounds", "celsius", "meters", "liters", "feet"], "to_u": ["miles", "km", "pounds", "kg", "fahrenheit", "feet", "gallons", "meters"]}},
    {"tool": "web_search", "templates": [
        ("搜索一下{query}", {"query": "{query}"}),
        ("帮我查一下{query}的信息", {"query": "{query}"}),
        ("Search for {query}", {"query": "{query}"}),
    ], "data": {"query": ["人工智能", "Python教程", "minimind", "大语言模型", "深度学习", "机器学习入门", "transformer架构", "RLHF训练方法"]}},
    {"tool": "read_file", "templates": [
        ("读取文件 {path}", {"path": "{path}"}),
        ("帮我看看 {path} 里面有什么", {"path": "{path}"}),
        ("What's in the file {path}?", {"path": "{path}"}),
    ], "data": {"path": ["/home/user/readme.txt", "/home/user/data/config.json", "/home/user/notes/meeting.txt", "/home/user/data/sales.csv"]}},
    {"tool": "sql_query", "templates": [
        ("查询{db}数据库中{cond}", {"database": "{db}", "query": "{sql}"}),
        ("从{db}中找出{cond}", {"database": "{db}", "query": "{sql}"}),
    ], "data": {"db": ["sales", "employees"], "sql": ["SELECT * FROM table", "SELECT * FROM table WHERE price > 1000", "SELECT COUNT(*) FROM table"], "cond": ["所有数据", "价格大于1000的记录", "总记录数"]}},
    {"tool": "get_stock_price", "templates": [
        ("查一下{symbol}的股价", {"symbol": "{symbol}"}),
        ("{symbol}现在多少钱？", {"symbol": "{symbol}"}),
        ("What's the price of {symbol}?", {"symbol": "{symbol}"}),
    ], "data": {"symbol": ["AAPL", "GOOGL", "MSFT", "600519", "TSLA", "AMZN"]}},
    {"tool": "get_route", "templates": [
        ("从{origin}到{destination}怎么走？", {"origin": "{origin}", "destination": "{destination}"}),
        ("帮我规划从{origin}到{destination}的路线", {"origin": "{origin}", "destination": "{destination}"}),
        ("How to get from {origin} to {destination}?", {"origin": "{origin}", "destination": "{destination}"}),
    ], "data": {"origin": ["北京", "上海", "New York"], "destination": ["上海", "杭州", "广州", "Los Angeles"]}},
    {"tool": "check_schedule", "templates": [
        ("{date}有什么日程？", {"date": "{date}"}),
        ("查看{date}的安排", {"date": "{date}"}),
        ("What's on my schedule for {date}?", {"date": "{date}"}),
    ], "data": {"date": ["2025-03-07", "2025-03-08", "2025-03-09"]}},
    {"tool": "create_event", "templates": [
        ("帮我创建一个日程：{title}，{start}到{end}", {"title": "{title}", "start_time": "{start}", "end_time": "{end}"}),
        ("安排一个会议：{title}，时间{start}-{end}", {"title": "{title}", "start_time": "{start}", "end_time": "{end}"}),
    ], "data": {"title": ["项目讨论", "代码审查", "需求评审", "技术分享", "周会"], "start": ["2025-03-10 09:00", "2025-03-10 14:00", "2025-03-11 10:00"], "end": ["2025-03-10 10:00", "2025-03-10 15:00", "2025-03-11 11:30"]}},
    {"tool": "summarize_text", "templates": [
        ("帮我总结一下这段文字：{text}", {"text": "{text}"}),
        ("Summarize this: {text}", {"text": "{text}"}),
    ], "data": {"text": ["人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。近年来，深度学习和大语言模型的发展推动了AI技术的快速进步。", "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。它以简洁清晰的语法和丰富的库生态而闻名，被广泛应用于Web开发、数据分析、人工智能等领域。"]}},
    {"tool": "random_number", "templates": [
        ("帮我生成一个{min}到{max}的随机数", {"min": "{min}", "max": "{max}"}),
        ("Generate a random number between {min} and {max}", {"min": "{min}", "max": "{max}"}),
    ], "data": {"min": [0, 1, 10, 100], "max": [100, 50, 999, 1000]}},
    {"tool": "text_length", "templates": [
        ("统计一下这段文字的长度：{text}", {"text": "{text}"}),
        ("How many characters in: {text}", {"text": "{text}"}),
    ], "data": {"text": ["你好世界", "Hello World", "人工智能正在改变世界", "The quick brown fox jumps over the lazy dog"]}},
    {"tool": "send_email", "templates": [
        ("给{to}发一封邮件，主题是{subject}，内容是{body}", {"to": "{to}", "subject": "{subject}", "body": "{body}"}),
        ("Send an email to {to} about {subject}", {"to": "{to}", "subject": "{subject}", "body": "{body}"}),
    ], "data": {"to": ["zhangsan@example.com", "lisi@company.com", "team@group.org"], "subject": ["项目进度更新", "会议通知", "代码审查请求", "Weekly Report"], "body": ["本周项目进展顺利，已完成80%的开发工作。", "请参加明天下午2点的技术评审会议。", "请帮忙审查PR #123的代码变更。"]}},
    {"tool": "get_location_info", "templates": [
        ("告诉我关于{name}的信息", {"name": "{name}"}),
        ("{name}在哪里？经纬度是多少？", {"name": "{name}"}),
        ("Tell me about {name}", {"name": "{name}"}),
    ], "data": {"name": ["北京", "上海", "New York", "Tokyo", "London"]}},
    {"tool": "list_directory", "templates": [
        ("列出 {path} 目录下的文件", {"path": "{path}"}),
        ("What files are in {path}?", {"path": "{path}"}),
    ], "data": {"path": ["/home/user", "/home/user/data", "/home/user/notes"]}},
    {"tool": "python_exec", "templates": [
        ("帮我运行这段Python代码：{code}", {"code": "{code}"}),
        ("Execute this code: {code}", {"code": "{code}"}),
    ], "data": {"code": ["print('Hello World')", "import math; print(math.pi)", "x = [1,2,3]; print(sum(x))", "print(2**10)"]}},
    {"tool": "countdown_timer", "templates": [
        ("帮我设置一个{seconds}秒的倒计时", {"duration_seconds": "{seconds}"}),
        ("Set a timer for {seconds} seconds", {"duration_seconds": "{seconds}"}),
    ], "data": {"seconds": [30, 60, 120, 300, 600]}},
]

MULTI_TURN_TEMPLATES = [
    {
        "name": "weather_then_compare",
        "tools_needed": ["get_current_weather", "get_current_weather"],
        "user_prompt": "帮我查一下{city1}和{city2}的天气，然后告诉我哪个城市更热",
        "data": {"city1": ["北京", "上海", "广州"], "city2": ["上海", "杭州", "深圳"]},
        "steps": [
            {"tool": "get_current_weather", "args_fn": lambda d: {"location": d["city1"]}},
            {"tool": "get_current_weather", "args_fn": lambda d: {"location": d["city2"]}},
        ],
        "gt_fn": lambda d: f"比较{d['city1']}和{d['city2']}的温度",
    },
    {
        "name": "exchange_then_calculate",
        "tools_needed": ["get_exchange_rate", "calculate_math"],
        "user_prompt": "我有{amount}美元，想换成人民币，能换多少？",
        "data": {"amount": [100, 500, 1000, 5000]},
        "steps": [
            {"tool": "get_exchange_rate", "args_fn": lambda d: {"from_currency": "USD", "to_currency": "CNY"}},
            {"tool": "calculate_math", "args_fn": lambda d: {"expression": f"{d['amount']}*7.21"}},
        ],
        "gt_fn": lambda d: f"{d['amount']}美元兑换人民币的计算结果",
    },
    {
        "name": "weather_and_route",
        "tools_needed": ["get_current_weather", "get_route"],
        "user_prompt": "查一下{city1}的天气，然后告诉我从{city1}到{city2}怎么走",
        "data": {"city1": ["北京", "上海"], "city2": ["上海", "杭州", "广州"]},
        "steps": [
            {"tool": "get_current_weather", "args_fn": lambda d: {"location": d["city1"]}},
            {"tool": "get_route", "args_fn": lambda d: {"origin": d["city1"], "destination": d["city2"]}},
        ],
        "gt_fn": lambda d: f"{d['city1']}的天气和到{d['city2']}的路线",
    },
    {
        "name": "schedule_and_create_event",
        "tools_needed": ["check_schedule", "create_event"],
        "user_prompt": "看看{date}有什么安排，然后帮我加一个'{event_title}'的会议，从{start}到{end}",
        "data": {"date": ["2025-03-07", "2025-03-08"], "event_title": ["新需求讨论", "技术方案评审"], "start": ["2025-03-07 16:00", "2025-03-08 14:00"], "end": ["2025-03-07 17:00", "2025-03-08 15:30"]},
        "steps": [
            {"tool": "check_schedule", "args_fn": lambda d: {"date": d["date"]}},
            {"tool": "create_event", "args_fn": lambda d: {"title": d["event_title"], "start_time": d["start"], "end_time": d["end"]}},
        ],
        "gt_fn": lambda d: f"查看{d['date']}日程并创建'{d['event_title']}'事件",
    },
    {
        "name": "search_and_summarize",
        "tools_needed": ["web_search", "summarize_text"],
        "user_prompt": "搜索一下{query}，然后帮我总结搜索结果",
        "data": {"query": ["人工智能", "大语言模型", "深度学习", "transformer"]},
        "steps": [
            {"tool": "web_search", "args_fn": lambda d: {"query": d["query"]}},
            {"tool": "summarize_text", "args_fn": lambda d: {"text": f"关于{d['query']}的搜索结果"}},
        ],
        "gt_fn": lambda d: f"搜索并总结关于{d['query']}的信息",
    },
    {
        "name": "translate_and_length",
        "tools_needed": ["translate_text", "text_length"],
        "user_prompt": "把'{text}'翻译成{lang}，然后告诉我翻译后有多少个字符",
        "data": {"text": ["你好世界", "今天天气真好", "机器学习很有趣"], "lang": ["english", "french", "japanese"]},
        "steps": [
            {"tool": "translate_text", "args_fn": lambda d: {"text": d["text"], "target_language": d["lang"]}},
            {"tool": "text_length", "args_fn": lambda d: {"text": "translated result"}},
        ],
        "gt_fn": lambda d: f"翻译'{d['text']}'到{d['lang']}并统计字符数",
    },
    {
        "name": "stock_and_exchange",
        "tools_needed": ["get_stock_price", "get_exchange_rate"],
        "user_prompt": "查一下{symbol}的股价，然后把价格从美元换算成人民币",
        "data": {"symbol": ["AAPL", "GOOGL", "MSFT", "TSLA"]},
        "steps": [
            {"tool": "get_stock_price", "args_fn": lambda d: {"symbol": d["symbol"], "market": "US"}},
            {"tool": "get_exchange_rate", "args_fn": lambda d: {"from_currency": "USD", "to_currency": "CNY"}},
        ],
        "gt_fn": lambda d: f"{d['symbol']}股价及美元转人民币",
    },
    {
        "name": "time_and_weather_multi_city",
        "tools_needed": ["get_current_time", "get_current_weather", "get_current_weather"],
        "user_prompt": "现在几点了？顺便查一下{city1}和{city2}的天气",
        "data": {"city1": ["北京", "上海"], "city2": ["Tokyo", "New York"]},
        "steps": [
            {"tool": "get_current_time", "args_fn": lambda d: {}},
            {"tool": "get_current_weather", "args_fn": lambda d: {"location": d["city1"]}},
            {"tool": "get_current_weather", "args_fn": lambda d: {"location": d["city2"]}},
        ],
        "gt_fn": lambda d: f"当前时间和{d['city1']}、{d['city2']}的天气",
    },
    {
        "name": "read_and_summarize",
        "tools_needed": ["read_file", "summarize_text"],
        "user_prompt": "帮我读取{path}文件，然后总结一下内容",
        "data": {"path": ["/home/user/readme.txt", "/home/user/notes/meeting.txt"]},
        "steps": [
            {"tool": "read_file", "args_fn": lambda d: {"path": d["path"]}},
            {"tool": "summarize_text", "args_fn": lambda d: {"text": "file content"}},
        ],
        "gt_fn": lambda d: f"读取并总结{d['path']}的内容",
    },
    {
        "name": "random_and_calculate",
        "tools_needed": ["random_number", "calculate_math"],
        "user_prompt": "帮我生成一个{min}到{max}的随机数，然后计算它的平方",
        "data": {"min": [1, 10, 100], "max": [100, 999, 1000]},
        "steps": [
            {"tool": "random_number", "args_fn": lambda d: {"min": d["min"], "max": d["max"]}},
            {"tool": "calculate_math", "args_fn": lambda d: {"expression": "random_number ** 2"}},
        ],
        "gt_fn": lambda d: f"生成{d['min']}-{d['max']}随机数并计算平方",
    },
]


def fill_template(template, data_values):
    result = template
    for key, val in data_values.items():
        result = result.replace("{" + key + "}", str(val))
    return result


def generate_single_turn_sample(template_info, available_tools):
    tool_name = template_info["tool"]
    tmpl = random.choice(template_info["templates"])
    user_template, arg_mapping = tmpl

    data_values = {}
    for key, val_template in arg_mapping.items():
        if val_template.startswith("{") and val_template.endswith("}"):
            data_key = val_template[1:-1]
            if data_key in template_info.get("data", {}):
                data_values[key] = random.choice(template_info["data"][data_key])
            else:
                data_values[key] = val_template
        else:
            data_values[key] = val_template

    user_prompt = fill_template(user_template, data_values)

    tool_def = TOOL_MAP.get(tool_name)
    if not tool_def:
        return None

    tool_args = {}
    for key, val in data_values.items():
        try:
            if key in ("value", "min", "max", "count", "duration_seconds", "limit", "timeout", "num_results"):
                tool_args[key] = int(val) if isinstance(val, (int, float)) or str(val).isdigit() else val
            else:
                tool_args[key] = str(val)
        except (ValueError, TypeError):
            tool_args[key] = str(val)

    tool_result = execute_tool(tool_name, tool_args)
    if tool_result is None:
        tool_result = {"result": "执行成功"}

    tool_call_content = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
    tool_result_str = json.dumps(tool_result, ensure_ascii=False)[:2048]

    gt = str(tool_result.get("result", tool_result.get("translated_text", tool_result.get("datetime", str(tool_result)))))

    selected_tools = random.sample(available_tools, min(random.randint(2, 5), len(available_tools)))
    if tool_name not in [t["function"]["name"] for t in selected_tools]:
        selected_tools.append(tool_def)

    conversations = [
        {"role": "system", "content": "你是一个有用的AI助手，可以使用提供的工具来帮助用户解决问题。", "tools": json.dumps(selected_tools, ensure_ascii=False)},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": tool_call_content},
        {"role": "tool", "content": tool_result_str},
        {"role": "assistant", "content": gt},
    ]

    return {"conversations": conversations, "gt": gt}


def generate_multi_turn_sample(template_info, available_tools):
    data_values = {}
    for key, options in template_info["data"].items():
        data_values[key] = random.choice(options)

    user_prompt = fill_template(template_info["user_prompt"], data_values)
    gt = template_info["gt_fn"](data_values)

    needed_tool_names = set()
    for step in template_info["steps"]:
        needed_tool_names.add(step["tool"])
    selected_tools = [TOOL_MAP[n] for n in needed_tool_names if n in TOOL_MAP]
    extra_tools = [t for t in available_tools if t["function"]["name"] not in needed_tool_names]
    if extra_tools:
        selected_tools.extend(random.sample(extra_tools, min(random.randint(1, 3), len(extra_tools))))

    conversations = [
        {"role": "system", "content": "你是一个有用的AI助手，可以使用提供的工具来帮助用户解决问题。请按步骤调用工具来完成任务。", "tools": json.dumps(selected_tools, ensure_ascii=False)},
        {"role": "user", "content": user_prompt},
    ]

    for step in template_info["steps"]:
        tool_name = step["tool"]
        tool_args = step["args_fn"](data_values)
        tool_result = execute_tool(tool_name, tool_args)
        if tool_result is None:
            tool_result = {"result": "执行成功"}

        tool_call_content = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False)
        tool_result_str = json.dumps(tool_result, ensure_ascii=False)[:2048]

        conversations.append({"role": "assistant", "content": tool_call_content})
        conversations.append({"role": "tool", "content": tool_result_str})

    conversations.append({"role": "assistant", "content": gt})

    return {"conversations": conversations, "gt": gt}


def generate_no_tool_samples(count):
    samples = []
    qa_pairs = [
        ("你好，你是谁？", "你好！我是MiniMind，一个小巧但有用的AI助手。"),
        ("1+1等于几？", "1+1等于2。"),
        ("中国的首都是哪里？", "中国的首都是北京。"),
        ("What is your name?", "My name is MiniMind, a lightweight AI assistant."),
        ("写一首短诗", "春风拂面来，花开满山坡。\n鸟语声声脆，人间好时节。"),
        ("什么是人工智能？", "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统，包括学习、推理、感知和决策等能力。"),
        ("推荐一本编程书", "推荐《Python编程：从入门到实践》，适合初学者系统学习Python编程。"),
        ("How are you?", "I'm doing well, thank you for asking! How can I help you today?"),
        ("今天星期几？", "我是一个AI，无法获取实时日期信息。请查看您的设备日历。"),
        ("讲个笑话", "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25（八进制31等于十进制25）。"),
    ]
    for _ in range(count):
        q, a = random.choice(qa_pairs)
        conversations = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        samples.append({"conversations": conversations, "gt": a})
    return samples


def main():
    parser = argparse.ArgumentParser(description="生成 Agent RL 训练数据")
    parser.add_argument("--output", type=str, default=os.path.join(DATA_DIR, "agent_rl.jsonl"), help="输出文件路径")
    parser.add_argument("--num_samples", type=int, default=10000, help="生成样本总数")
    parser.add_argument("--multi_turn_ratio", type=float, default=0.4, help="多轮对话比例")
    parser.add_argument("--no_tool_ratio", type=float, default=0.1, help="无需工具的对话比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    num_multi = int(args.num_samples * args.multi_turn_ratio)
    num_no_tool = int(args.num_samples * args.no_tool_ratio)
    num_single = args.num_samples - num_multi - num_no_tool

    available_tools = TOOLS
    all_samples = []

    print(f"=== Agent RL 数据生成 ===")
    print(f"总样本数: {args.num_samples}")
    print(f"  单轮工具调用: {num_single}")
    print(f"  多轮工具调用: {num_multi}")
    print(f"  无工具对话: {num_no_tool}")

    print(f"\n[1/3] 生成 {num_single} 条单轮工具调用数据...")
    for i in range(num_single):
        template_info = random.choice(SINGLE_TURN_TEMPLATES)
        sample = generate_single_turn_sample(template_info, available_tools)
        if sample:
            all_samples.append(sample)
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{num_single}")

    print(f"\n[2/3] 生成 {num_multi} 条多轮工具调用数据...")
    for i in range(num_multi):
        template_info = random.choice(MULTI_TURN_TEMPLATES)
        sample = generate_multi_turn_sample(template_info, available_tools)
        if sample:
            all_samples.append(sample)
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{num_multi}")

    print(f"\n[3/3] 生成 {num_no_tool} 条无工具对话数据...")
    no_tool_samples = generate_no_tool_samples(num_no_tool)
    all_samples.extend(no_tool_samples)

    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n=== 生成完成 ===")
    print(f"总样本数: {len(all_samples)}")
    print(f"输出文件: {args.output}")
    print(f"文件大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
