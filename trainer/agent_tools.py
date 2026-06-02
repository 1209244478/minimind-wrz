"""
Agent 工具定义模块
统一管理所有工具定义、模拟数据、执行函数和参数校验
供 train_agent.py 和 eval_toolcall.py 共享使用
"""
import json
import math
import random
import re
import signal

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如 123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算，支持长度、重量、温度等常见单位", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如 km/miles/kg/pounds/celsius/fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius", "description": "温度单位"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如 Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如 USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如 english/chinese/japanese/french"}}, "required": ["text", "target_language"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "在互联网上搜索信息，返回最相关的结果摘要", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}, "num_results": {"type": "integer", "description": "返回结果数量", "default": 3, "minimum": 1, "maximum": 10}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取指定路径的文件内容", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "encoding": {"type": "string", "description": "文件编码", "default": "utf-8"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "将内容写入指定路径的文件", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "要写入的内容"}, "mode": {"type": "string", "enum": ["write", "append"], "default": "write", "description": "写入模式：覆盖或追加"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_directory", "description": "列出指定目录下的文件和子目录", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径"}, "pattern": {"type": "string", "description": "文件名过滤模式，如 *.txt", "default": "*"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "sql_query", "description": "执行 SQL 查询并返回结果", "parameters": {"type": "object", "properties": {"database": {"type": "string", "description": "数据库名称"}, "query": {"type": "string", "description": "SQL 查询语句"}, "limit": {"type": "integer", "description": "返回结果行数限制", "default": 100}}, "required": ["database", "query"]}}},
    {"type": "function", "function": {"name": "python_exec", "description": "执行 Python 代码并返回输出结果", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}, "timeout": {"type": "integer", "description": "超时时间（秒）", "default": 10}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "http_request", "description": "发送 HTTP 请求并返回响应", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "请求 URL"}, "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET", "description": "HTTP 方法"}, "headers": {"type": "object", "description": "请求头", "default": {}}, "body": {"type": "object", "description": "请求体（JSON）"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "发送电子邮件", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "收件人邮箱地址"}, "subject": {"type": "string", "description": "邮件主题"}, "body": {"type": "string", "description": "邮件正文"}, "cc": {"type": "string", "description": "抄送地址", "default": ""}}, "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "get_stock_price", "description": "查询股票实时价格和涨跌信息", "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "股票代码，如 AAPL、600519（贵州茅台）"}, "market": {"type": "string", "enum": ["US", "CN", "HK"], "default": "CN", "description": "市场：美国/中国/香港"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "get_route", "description": "获取两个地点之间的路线规划", "parameters": {"type": "object", "properties": {"origin": {"type": "string", "description": "出发地"}, "destination": {"type": "string", "description": "目的地"}, "mode": {"type": "string", "enum": ["driving", "transit", "walking"], "default": "driving", "description": "出行方式：驾车/公交/步行"}}, "required": ["origin", "destination"]}}},
    {"type": "function", "function": {"name": "create_event", "description": "创建日程事件", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "事件标题"}, "start_time": {"type": "string", "description": "开始时间，如 2025-03-07 14:00"}, "end_time": {"type": "string", "description": "结束时间"}, "location": {"type": "string", "description": "地点", "default": ""}, "description": {"type": "string", "description": "事件描述", "default": ""}}, "required": ["title", "start_time", "end_time"]}}},
    {"type": "function", "function": {"name": "check_schedule", "description": "查询指定日期的日程安排", "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "日期，如 2025-03-07"}, "timezone": {"type": "string", "description": "时区", "default": "Asia/Shanghai"}}, "required": ["date"]}}},
    {"type": "function", "function": {"name": "summarize_text", "description": "对长文本进行摘要提取", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要摘要的文本"}, "max_length": {"type": "integer", "description": "摘要最大长度（字数）", "default": 200}, "style": {"type": "string", "enum": ["concise", "detailed", "bullet_points"], "default": "concise", "description": "摘要风格：简洁/详细/要点"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}, "count": {"type": "integer", "description": "生成数量", "default": 1, "minimum": 1, "maximum": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "get_location_info", "description": "获取地点的详细信息，包括经纬度、时区、人口等", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "地点名称"}, "info_type": {"type": "string", "enum": ["basic", "detailed"], "default": "basic", "description": "信息详细程度"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "countdown_timer", "description": "设置倒计时器", "parameters": {"type": "object", "properties": {"duration_seconds": {"type": "integer", "description": "倒计时秒数"}, "label": {"type": "string", "description": "计时器标签", "default": "timer"}}, "required": ["duration_seconds"]}}},
]

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

WEATHER_DATA = {
    "北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"),
    "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"),
    "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"),
    "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
    "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗"),
    "Los Angeles": ("22°C", "晴"), "Singapore": ("31°C", "雷阵雨"), "Seoul": ("10°C", "多云"),
}

TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00",
    "Asia/Singapore": "2025-03-07 14:30:00", "Asia/Seoul": "2025-03-07 15:30:00",
}

EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79,
    ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72, ("KRW", "CNY"): 0.0053,
    ("SGD", "CNY"): 5.35, ("HKD", "CNY"): 0.92,
}

TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today",
    ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting",
    ("Happy birthday", "chinese"): "生日快乐", ("谢谢", "japanese"): "ありがとう",
    ("Hello", "french"): "Bonjour", ("再见", "english"): "Goodbye",
    ("How are you", "chinese"): "你好吗",
}

UNIT_DATA = {
    "km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462,
    "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048,
    "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556,
    "liters_gallons": 0.264172, "gallons_liters": 3.78541,
}

STOCK_DATA = {
    "AAPL": ("$178.50", "+1.2%"), "GOOGL": ("$142.30", "-0.5%"),
    "MSFT": ("$415.80", "+0.8%"), "600519": ("¥1,756.00", "+0.3%"),
    "000001": ("¥12.45", "-0.2%"), "0700": ("HK$352.40", "+1.5%"),
    "TSLA": ("$245.60", "-1.1%"), "AMZN": ("$178.90", "+0.6%"),
}

ROUTE_DATA = {
    ("北京", "上海"): {"driving": "1,213km, 约12小时", "transit": "高铁约4.5小时", "walking": "不推荐"},
    ("北京", "广州"): {"driving": "2,130km, 约22小时", "transit": "高铁约8小时", "walking": "不推荐"},
    ("上海", "杭州"): {"driving": "180km, 约2小时", "transit": "高铁约1小时", "walking": "约36小时"},
    ("New York", "Los Angeles"): {"driving": "4,501km, 约42小时", "transit": "飞机约5.5小时", "walking": "不推荐"},
}

SCHEDULE_DATA = {
    "2025-03-07": [
        {"title": "团队周会", "start": "09:00", "end": "10:00", "location": "会议室A"},
        {"title": "项目评审", "start": "14:00", "end": "15:30", "location": "线上"},
        {"title": "午餐", "start": "12:00", "end": "13:00", "location": "公司食堂"},
    ],
    "2025-03-08": [{"title": "技术分享", "start": "10:00", "end": "11:30", "location": "会议室B"}],
    "2025-03-09": [],
}

LOCATION_DATA = {
    "北京": {"lat": 39.9042, "lng": 116.4074, "timezone": "Asia/Shanghai", "population": "2171万", "country": "中国"},
    "上海": {"lat": 31.2304, "lng": 121.4737, "timezone": "Asia/Shanghai", "population": "2489万", "country": "中国"},
    "New York": {"lat": 40.7128, "lng": -74.0060, "timezone": "America/New_York", "population": "841万", "country": "美国"},
    "Tokyo": {"lat": 35.6762, "lng": 139.6503, "timezone": "Asia/Tokyo", "population": "1396万", "country": "日本"},
    "London": {"lat": 51.5074, "lng": -0.1278, "timezone": "Europe/London", "population": "898万", "country": "英国"},
}

FILE_SYSTEM = {
    "/home/user/readme.txt": "这是一个示例项目。\n版本：1.0.0\n作者：MiniMind Team",
    "/home/user/data/config.json": '{"model": "minimind", "version": "5", "hidden_size": 1024}',
    "/home/user/notes/meeting.txt": "2025-03-07 会议记录\n1. 讨论了新版本发布计划\n2. 确定了性能优化目标\n3. 下周进行代码审查",
    "/home/user/data/sales.csv": "date,product,amount\n2025-01,A,1000\n2025-02,B,1500\n2025-03,A,1200",
}

SQL_DATABASES = {
    "sales": [
        {"id": 1, "product": "笔记本电脑", "price": 5999, "quantity": 120},
        {"id": 2, "product": "手机", "price": 3999, "quantity": 350},
        {"id": 3, "product": "耳机", "price": 299, "quantity": 800},
    ],
    "employees": [
        {"id": 1, "name": "张三", "department": "技术部", "salary": 25000},
        {"id": 2, "name": "李四", "department": "市场部", "salary": 18000},
        {"id": 3, "name": "王五", "department": "技术部", "salary": 30000},
    ],
}

WEB_SEARCH_RESULTS = {
    "人工智能": [
        {"title": "人工智能 - 维基百科", "snippet": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"},
        {"title": "AI最新进展", "snippet": "2025年大语言模型技术持续突破，多模态AI成为主流趋势。"},
    ],
    "Python教程": [
        {"title": "Python官方教程", "snippet": "Python是一种解释型、高级、通用编程语言，由Guido van Rossum创建。"},
        {"title": "Python入门指南", "snippet": "从零开始学习Python，包括基础语法、数据结构、面向对象编程等。"},
    ],
    "minimind": [
        {"title": "MiniMind - 最小大模型", "snippet": "MiniMind是一个开源的小型语言模型项目，旨在用最小参数实现可用智能。"},
    ],
}

def _get_location_info(args):
    name = args.get("name", "")
    info = LOCATION_DATA.get(name)
    if not info:
        return {"error": "地点未找到"}
    result = {"name": name, "lat": info["lat"], "lng": info["lng"], "country": info["country"]}
    if args.get("info_type", "basic") == "detailed":
        result.update({"timezone": info["timezone"], "population": info["population"]})
    return result


MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
    "web_search": lambda args: {"results": WEB_SEARCH_RESULTS.get(args.get("query", ""), [{"title": "搜索结果", "snippet": f"关于'{args.get('query', '')}'的搜索结果"}])[:args.get("num_results", 3)]},
    "read_file": lambda args: {"content": FILE_SYSTEM.get(args.get("path", ""), "文件不存在或无法读取"), "path": args.get("path", "")},
    "write_file": lambda args: {"status": "success", "path": args.get("path", ""), "bytes_written": len(args.get("content", ""))},
    "list_directory": lambda args: {"path": args.get("path", ""), "entries": ["readme.txt", "data/", "notes/", "config.json"]},
    "sql_query": lambda args: {"results": SQL_DATABASES.get(args.get("database", ""), [])[:args.get("limit", 100)], "row_count": len(SQL_DATABASES.get(args.get("database", ""), [])), "database": args.get("database", "")},
    "python_exec": lambda args: {"output": "代码执行成功", "exit_code": 0},
    "http_request": lambda args: {"status_code": 200, "body": {"message": "OK"}, "headers": {"content-type": "application/json"}},
    "send_email": lambda args: {"status": "sent", "to": args.get("to", ""), "subject": args.get("subject", "")},
    "get_stock_price": lambda args: (lambda s: {"symbol": args.get("symbol", ""), "price": s[0], "change": s[1], "market": args.get("market", "CN")})(STOCK_DATA.get(args.get("symbol", ""), ("N/A", "0%"))),
    "get_route": lambda args: (lambda r: {"origin": args.get("origin", ""), "destination": args.get("destination", ""), "mode": args.get("mode", "driving"), "info": r.get(args.get("mode", "driving"), "路线信息暂不可用")})(ROUTE_DATA.get((args.get("origin", ""), args.get("destination", "")), {"driving": "路线信息暂不可用", "transit": "路线信息暂不可用", "walking": "路线信息暂不可用"})),
    "create_event": lambda args: {"status": "created", "title": args.get("title", ""), "start_time": args.get("start_time", ""), "end_time": args.get("end_time", "")},
    "check_schedule": lambda args: {"date": args.get("date", ""), "events": SCHEDULE_DATA.get(args.get("date", ""), [])},
    "summarize_text": lambda args: {"summary": args.get("text", "")[:min(args.get("max_length", 200), len(args.get("text", "")))] + ("..." if len(args.get("text", "")) > args.get("max_length", 200) else ""), "original_length": len(args.get("text", "")), "summary_length": min(args.get("max_length", 200), len(args.get("text", "")))},
    "random_number": lambda args: {"numbers": [random.randint(int(args.get("min", 0)), int(args.get("max", 100))) for _ in range(args.get("count", 1))], "min": args.get("min", 0), "max": args.get("max", 100)},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split()), "chinese_chars": sum(1 for c in args.get("text", "") if '\u4e00' <= c <= '\u9fff')},
    "get_location_info": lambda args: _get_location_info(args),
    "countdown_timer": lambda args: {"status": "started", "duration_seconds": args.get("duration_seconds", 0), "label": args.get("label", "timer")},
}

CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and a.get("target_language"),
    "web_search": lambda a: bool(a.get("query")),
    "read_file": lambda a: bool(a.get("path")),
    "write_file": lambda a: bool(a.get("path")) and a.get("content") is not None,
    "list_directory": lambda a: bool(a.get("path")),
    "sql_query": lambda a: bool(a.get("database")) and bool(a.get("query")),
    "python_exec": lambda a: bool(a.get("code")),
    "http_request": lambda a: bool(a.get("url")),
    "send_email": lambda a: bool(a.get("to")) and bool(a.get("subject")) and bool(a.get("body")),
    "get_stock_price": lambda a: bool(a.get("symbol")),
    "get_route": lambda a: bool(a.get("origin")) and bool(a.get("destination")),
    "create_event": lambda a: bool(a.get("title")) and bool(a.get("start_time")) and bool(a.get("end_time")),
    "check_schedule": lambda a: bool(a.get("date")),
    "summarize_text": lambda a: bool(a.get("text")),
    "random_number": lambda a: True,
    "text_length": lambda a: bool(a.get("text")),
    "get_location_info": lambda a: bool(a.get("name")),
    "countdown_timer": lambda a: a.get("duration_seconds") is not None and a.get("duration_seconds", 0) > 0,
}


def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'【(.*?)】', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    for m in re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL):
        try:
            data = json.loads(m.strip())
            if isinstance(data, dict) and data.get("name"):
                calls.append(data)
        except Exception:
            pass
    return calls


def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(1)
        return fn(args)
    except Exception:
        return None
    finally:
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        except Exception:
            pass


def get_tools(names):
    return [TOOL_MAP[n] for n in names if n in TOOL_MAP]


REACT_SYSTEM_PROMPT = """你是一个有用的AI助手，可以使用工具来帮助用户解决问题。

请按照 ReAct (Reasoning + Acting) 模式工作：
1. **Thought**: 分析当前情况，思考下一步该做什么
2. **Action**: 调用合适的工具（使用【】格式输出JSON）
3. **Observation**: 观察工具返回的结果
4. 重复以上步骤直到可以给出最终答案
5. **Answer**: 基于所有观察结果给出最终回答

示例：
Thought: 用户想知道北京天气，我需要调用天气查询工具
【{"name": "get_current_weather", "arguments": {"location": "北京"}}】
Observation: 北京当前温度28°C，晴天
Answer: 北京今天天气晴朗，气温28°C。"""

PLAN_EXECUTE_SYSTEM_PROMPT = """你是一个有用的AI助手，可以使用工具来帮助用户解决问题。

请按照 Plan-Execute 模式工作：
1. **Plan**: 制定完成任务的计划，列出需要执行的步骤
2. **Execute**: 按计划逐步执行，调用需要的工具（使用【】格式输出JSON）
3. **Evaluate**: 评估执行结果是否满足需求
4. 如果计划未完成，继续执行下一步；如果已完成，给出最终回答

示例：
Plan: 
1. 查询北京天气
2. 查询上海天气
3. 比较两个城市的温度

Execute Step 1:
【{"name": "get_current_weather", "arguments": {"location": "北京"}}】
Result: 北京28°C，晴

Execute Step 2:
【{"name": "get_current_weather", "arguments": {"location": "上海"}}】
Result: 上海15°C，多云

Evaluate: 已获取两个城市的天气数据，可以进行比较
Answer: 北京28°C比上海15°C更热，北京比上海高13°C。"""

SYSTEM_PROMPTS = {
    "default": "你是一个有用的AI助手，可以使用提供的工具来帮助用户解决问题。",
    "react": REACT_SYSTEM_PROMPT,
    "plan_execute": PLAN_EXECUTE_SYSTEM_PROMPT,
}
