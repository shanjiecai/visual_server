import cv2
import time
import json
import requests
from collections import defaultdict
from typing import List, Dict
import base64
import re
import os
import threading
from threading import Thread
from ultralytics import YOLOE
# import supervision as sv
import openai
from flask import Flask, request, jsonify, Response
import numpy as np
import logging
# from flask_cors import CORS
import queue
from openai import OpenAI
# 添加CLIP相关导入
import torch
from PIL import Image
from tqdm import tqdm
import shutil
from transformers import CLIPProcessor, CLIPModel

# ===========================
# 模拟大模型接口（视觉识别 + 问答）
# 实际情况需替换为你自己的大模型 API 调用
# ===========================
URL_7B = "http://36.156.143.242:8090/qwenvl2-7b"
URL_72B = "http://36.156.143.242:8090/describe-pic"

MAX_PROCESSING_TIMEOUT = 10*60  # 创建任务后，最长等待时间（秒），如果期间没有等到后续请求，则任务自动删除
openai.api_key = ""
client = openai.OpenAI(
    base_url="http://10.112.0.32:5239/v1",
    api_key="",
)
client_visual = openai.OpenAI(
    base_url="http://cc.komect.com/llm/vlgroup/",
    api_key="EMPTY",
)
# models = client_visual.models.list()
# model = models.data[0].id
processing_pool = dict()
os.environ['PYTHONHASHSEED'] = '0'
app = Flask(__name__)
# CORS(app)  # 允许所有域名跨域请求（也可以设置 origins）


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def chat_gpt(message):
    response = client.chat.completions.create(
        model="qwen2.5-7b-test",  # qwen72 本次支持模型
        stream=False,  # 如果需要流式响应，请设置为 True
        messages=message  # 输入的用户消息
    )

    # 输出模型的非流式响应
    # for chunk in response._iter_events():
    #     content = chunk.data
    #     if content:
    #         print(content, end="\n", flush=True)
    return response


names = ['dining table', 'four-poster bed', 'Refrigerator', 'microwave oven', 'television', 'washer', 'cup',
         'paper knife', 'wooden spoon', 'flower',
         'remote control', 'comic book', 'mobile phone', 'hand-held computer', 'desktop computer', 'computer keyboard',
         'dishwasher', 'person', 'ruler', 'tandem bicycle']

# MODEL = YOLOE("pretrain/yoloe-v8l-seg.pt")
# MODEL.to("cuda:0")
# MODEL.set_classes(names, MODEL.get_text_pe(names))


thickness = 0
def extract_categories_from_question(class_dict: List[str], message: str) -> List[str]:
    formatted = [f"{i}-{item}" for i, item in enumerate(class_dict)]

    """
    从自然语言问题中提取关键词类别
    """
    # 构造 prompt
    prompt = f""" ## Background ##
    你是一位视觉语言理解专家。用户提出了一个问题，你需要根据语义，在以下类目中找出最相关的一项或多项。

    即使用户没有直接提到类目的名字，也请结合含义判断是否相关。

    请从类目列表中返回最相关的项（如"水杯"和"手机"等）在列表中的位置，不必解释原因，不能返回不在列表中的项。
    """
    query  = message[-1]["content"]

    messages = [
        {'role': 'system', 'content': prompt},
        {
        "role": "user",
        "content": 
            f"问题：{query},从提供类目列表{formatted}中返回最相关的项在列表中的位置，不必解释原因。"
        }
    ]
    print(messages)

   
    # 调用大模型
    try:
        t1 = time.time()
        response = chat_gpt(messages)
        t2 = time.time()
        print(f"语言大模型请求完成，耗时: {t2 - t1:.2f}秒")

       
        print(response.response)
        keywords = response.response.split(",")
           
        return keywords
       
        
    except Exception as e:
        print(f"API调用异常: {e}")
        return []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.str_, np.unicode_)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class QwenLLM(object):
    def __init__(
            self,
            url="http://10.112.0.32:5239/v1",
            backup_url="http://127.0.0.1:5020/v1/",
            # model_type="qwen14",
            model_type="qwen72",
            api_key="sk-tbXDmYh3m2VQgdA7931a909a19E24cBeAc6cC59783521969",
            max_tokens=8000, temperature=0.8, top_k=3
    ):
        self.url = url
        self.backup_url = backup_url
        self.api_key = api_key

        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k

        self.llm_client = OpenAI(base_url=self.url, api_key=self.api_key)
        self.llm_client_bp = OpenAI(base_url=self.backup_url, api_key=self.api_key)

    def chat_stream(self, request_id, messages, do_search=False):
        start = time.time()

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_type,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=2,
                stream=True
            )
        except Exception as ee:
            print(str(ee))
            try:
                response = self.llm_client_bp.chat.completions.create(
                    model=self.model_type,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=2,
                    stream=True,
                    extra_body={"request_id": request_id}
                )
            except Exception as eee:
                print(str(eee))
                return None

        response.encoding = "utf-8"
        headers = {"Transfer-Encoding": "chunked", "Content-Type": "text/event-stream", "charset": "UTF-8"}
        output = self.stream_gen(response, do_search, start=start)
        # return Response(output, headers=headers, content_type="text/event-stream")
        return output

    def stream_gen(self, stream_resp, do_search, start):
        chunk_list = []
        for chunk in stream_resp._iter_events():
            try:
                answer_chunk = chunk.data
            except Exception:
                continue
            if not answer_chunk:
                continue
            if answer_chunk.strip() == "[DONE]":
                continue
            out_dict = {

                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": answer_chunk
                    }
                ]

            }
            # output_str = json.dumps(out_dict, ensure_ascii=False)
            # self.logger.info(output_str.strip())
            chunk_list.append(answer_chunk)
            yield out_dict


class LocalLLM(object):
    def __init__(
            self,
            url="http://10.112.0.32:5239/v1",
            model_type="deepseek-r1-kanjia",
            api_key="sk-TIq3HJsIBBF6UWnn65E6BbE0A8B4484bB7A5AeC4Df249f7c",
            max_tokens=10240, temperature=0.7, top_p=0.7, top_k=5
    ):
        self.url = url
        self.api_key = api_key

        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.llm_client = OpenAI(base_url=self.url, api_key=self.api_key)

    def chat(self, request_id, messages, do_search=False):
        start = time.time()

        try:
            content, reasoning_content, references = "", "", []
            response = self.llm_client.chat.completions.create(
                model=self.model_type,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=60,
                stream=False,
                extra_body={"request_id": request_id}
            )

            content = response.response.replace("\\n", "")
            if ("<think>" in content) and ("</think>" in content):
                content_1, str_2 = content.split("<think>")
                reasoning_content, content_2 = str_2.split("</think>")
                content = content_1 + content_2

            return content, reasoning_content, references
        except Exception as ee:
            return None

    def chat_stream(self, request_id, messages, do_search=False):
        start = time.time()

        try:
            # response = self.llm_client.chat.completions.create(
            #     model=self.model_type,
            #     messages=messages,
            #     max_tokens=self.max_tokens,
            #     temperature=self.temperature,
            #     timeout=2,
            #     stream=True,
            # )
            response = client.chat.completions.create(
                model="qwen2.5-7b-test",  # qwen72 本次支持模型
                stream=True,  # 如果需要流式响应，请设置为 True
                messages=messages  # 输入的用户消息
            )
            response = self.llm_client.chat.completions.create(
                model=self.model_type,
                messages=messages,
                temperature=self.temperature,
                extra_body={"top_k": self.top_k},
                timeout=2,
                stream=True
            )
        except Exception as ee:
            print(ee)
            return None

        response.encoding = "utf-8"

        output = self.stream_gen(response, do_search, start=start)
        # return Response(output, content_type="text/event-stream")
        return output

    def stream_gen(self, stream_resp, do_search, start):
        chunk_list = []
        for chunk in stream_resp._iter_events():
            print(chunk.data)
            try:
                answer_chunk = chunk.data
            except Exception:
                continue
            if not answer_chunk:
                continue
            if answer_chunk.strip() == "[DONE]":
                continue
            out_dict = {
                "return_code": 0, "return_msg": "Success",
                "response": {

                    "index": 0,
                    "finish_reason": None,
                    "delta": answer_chunk  # chunk["choices"][0]["message"]["content"]

                }
            }
            output_str = json.dumps(out_dict, ensure_ascii=False)
            # self.logger.info(output_str.strip())
            chunk_list.append(answer_chunk)
            yield output_str


class VisualLLM(object):
    def __init__(self, sn=None,
                 ):
        self.sn = sn
        if sn in processing_pool:
            self.processor = processing_pool[sn]
        else:
            logging.info(jsonify({'error': 'Stream not found in processing pool'}))

        result = self.processor.get_result()
        if result and 'class' in result and 'base64' in result:
            self.class_to_images = result['class']
            self.base64_to_id = result['base64']
            # 检查是否有足够的图像数据
            if not self.class_to_images or not self.base64_to_id:
                logging.info(jsonify({'error': 'No image data available for the stream'}))

    def chat_stream(self, request_id, messages, do_search=False):
        start = time.time()
        try:
            print("\nmessage：", messages)
            # 获取所有可用的类别
            available_categories = list(self.class_to_images.keys())
            categories = extract_categories_from_question(available_categories, messages)
            print("识别出的问题相关类别：", categories)

            # 收集所有相关类别的图片ID
            all_image_ids = set()
            for category in categories:
                if category in self.class_to_images:
                    all_image_ids.update(self.class_to_images[category])
            all_image_ids = set()

            category = available_categories[int(categories[0])]
            if category in self.class_to_images:
                all_image_ids.update(self.class_to_images[category])
            # 去重后的图片ID列表
            unique_image_ids = list(all_image_ids)
            print(f"找到 {len(unique_image_ids)} 张相关图片")

            if unique_image_ids:
                # 获取这些图像ID对应的base64编码
                unique_image_ids = unique_image_ids[-1:]
                images_base64 = [self.base64_to_id[img_id] for img_id in unique_image_ids]
                # 将所有相关图片一起发送给模型
                output = self.stream_gen(request_id, ask_model_about_category(images_base64, messages, categories),
                                         do_search, start=start)
                return output
            else:
                print("\n未找到相关图片，无法回答问题。")
                return "未找到相关图片，无法回答问题。"
        except Exception as ee:
            print(str(ee))
            return None

        # return Response(output, content_type="text/event-stream")

    def stream_gen(self, request_id, stream_resp, do_search, start):
        chunk_list = []
        for chunk in stream_resp:

            if chunk:
                # decoded_line = chunk.decode("utf-8")

                # # 分割字符串，提取JSON部分
                # decoded_line = decoded_line.split('data: ', 1)[1].strip()
                # # 解析为字典
                # decoded_line = json.loads( decoded_line)
                if chunk.choices[0].finish_reason == "stop":

                    resp = {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "delta": chunk.choices[0].delta.content
                                # chunk["choices"][0]["message"]["content"]
                            }
                        ]

                    }
                    break
                else:
                    resp = {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "null",
                                "delta": chunk.choices[0].delta.content  # chunk["choices"][0]["message"]["content"]
                            }
                        ]

                    }

                # self.logger.info(output_str.strip()
                chunk_list.append(resp)
                yield resp


class StreamThread(Thread):
    def __init__(self, name, logger, llm, request_id, messages, do_search, output_queue):
        super().__init__(daemon=True)
        self.name = name
        self.logger = logger
        self.llm = llm
        self.request_id = request_id
        self.messages = messages
        self.do_search = do_search
        self.output_queue = output_queue

    def run(self, ):
        try:
            response = self.llm.chat_stream(
                request_id=self.request_id, messages=self.messages, do_search=self.do_search)
            for chunk in response:
                # 将流式输出放入队列
                self.output_queue.put((self.name, chunk))

            self.output_queue.put((self.name, None))  # 标记此接口输出结束
        except Exception as e:
            print(e)
            self.output_queue.put((self.name, None))


class SearchLLM(object):
    def __init__(self, logger, foreword_llm, generator_llm):
        self.logger = logger
        self.foreword_llm = foreword_llm
        self.generator_llm = generator_llm

    def chat_stream(self, request_id, foreword_messages, generator_messages, do_search=False, server_start=False):
        def format_data(resp):
            return f"data:{json.dumps({'response': resp}, ensure_ascii=False)}\n\n"

        output_queue = queue.Queue(maxsize=0)
        # 创建引导语生成大模型线程
        foreword_thread = StreamThread(
            name="Foreword",
            logger=self.logger,
            llm=self.foreword_llm,
            request_id=request_id,
            messages=foreword_messages,
            do_search=do_search,
            output_queue=output_queue
        )
        # 创建联网搜索大模型线程
        generator_thread = StreamThread(
            name="Generator",
            logger=self.logger,
            llm=self.generator_llm,
            request_id=request_id,
            messages=generator_messages,
            do_search=do_search,
            output_queue=output_queue
        )
        generator_buffer = queue.Queue()  # 用于暂存【联网搜索大模型 Generator】流式输出

        # 启动线程
        foreword_thread.start()
        generator_thread.start()

        thread_name, thread_chunk = output_queue.get()
        # take_time = (time.time() - start) * 1000
        # self.logger.info("{} {:.2f}ms {}".format(thread_name, take_time, thread_chunk.strip()))

        first_thread_name = thread_name
        if first_thread_name == "Generator":
            # 【联网搜索大模型Generator】先返回，则直接输出
            while True:
                if thread_name == "Generator":
                    if thread_chunk is not None:
                        yield format_data(thread_chunk)
                    else:
                        break
                thread_name, thread_chunk = output_queue.get()
        else:  # 【引导语生成大模型 Foreword】先返回，则先输出引导语内容，再输出【联网搜索大模型Generator】
            fore_finished_flag = False
            gen_finished_flag = False
            while True:
                if thread_name == "Foreword":
                    if thread_chunk is not None:
                        yield format_data(thread_chunk)
                    else:
                        fore_finished_flag = True
                elif thread_name == "Generator":
                    if thread_chunk is not None:
                        generator_buffer.put(thread_chunk)
                        if fore_finished_flag:
                            while not generator_buffer.empty():
                                buffer_chunk = generator_buffer.get()
                                yield format_data(buffer_chunk)
                    else:
                        gen_finished_flag = True

                if fore_finished_flag and gen_finished_flag:
                    break
                thread_name, thread_chunk = output_queue.get()

        while not generator_buffer.empty():
            buffer_chunk = generator_buffer.get()
            yield format_data(buffer_chunk)


def process_image(image):
    # results = MODEL.predict(image, verbose=False)
    # detections = sv.Detections.from_ultralytics(results[0])
    # if (len(detections.xyxy) == 0):
    #     return []
    # else:
    #     labels = [
    #         [0, 0, 0, 0, class_name, confidence]
    #         for class_name, confidence
    #         in zip(detections["class_name"], detections.confidence)]
    #     for i in range(len(detections.xyxy)):
    #         labels[i][0:4] = detections.xyxy[i][0:4]
    return [0, 0, 0, 0, 0, 0]


def process_grounding(frame):
    # start_time = time.time()
    result = process_image(frame)

    return [grounding for grounding in result if grounding[5] > thickness]


def generator_head(iter_response, sessionId):
    logging.info("返回请求时间%s",
                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}")
    i = 0
    for line in iter_response.iter_lines():
        if line:
            try:
                decoded_line = line.decode("utf-8")

                # 分割字符串，提取JSON部分
                decoded_line = decoded_line.split('data: ', 1)[1].strip()
                # 解析为字典
                decoded_line = json.loads(decoded_line)
                if decoded_line["choices"][0]["finish_reason"] == "stop":

                    resp = {
                        "id": sessionId,
                        "choices": [{
                            "index": i,
                            "finish_reason": "stop",
                            "delta": decoded_line["choices"][0]["delta"]["content"]
                            # chunk["choices"][0]["message"]["content"]
                        }]
                    }
                    break
                else:
                    resp = {
                        "id": sessionId,
                        "choices": [{
                            "index": i,
                            "finish_reason": "null",
                            "delta": decoded_line["choices"][0]["delta"]["content"]
                            # chunk["choices"][0]["message"]["content"]

                        }]
                    }
                i = i + 1
                # yield f"data:{"response": {json.dumps(resp, ensure_ascii=False)}}\n\n"

                formatted_data = f"data:{json.dumps({'response': resp}, ensure_ascii=False)}\n\n"

                yield formatted_data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")


def ask_model_about_category(images_64: List[str], question: str,category:str) -> str:
    """
    调用大模型，分析指定图像中某类别的行为
    """
    # 构造 prompt

# EXTRACT_MEMORY_PROMPT = f"""
#     你是一位视觉记忆与推理专家，拥有连续多帧图像的观察结果。

#     请你根据以下连续拍摄的图像，分析用户的问题，并根据图像中的内容进行推理判断。
#     用户问题：
#     {question}

#     请你判断问题中的物体在图像中的表现、变化或位置，并给出直接明确的答案。

#     如果涉及空间位置，请只返回物体当前所在的位置描述（如"地上"、"桌子下"等），不要解释过程或编号。"""
    
    # messages = [
    #      {'role': 'system', 'content': SYS_MEMORY_PROMPT},
    #      {
    #         "role": "user",
    #         "content": [
    #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_64}"}},
    #             {"type": "text", "text":question},
    #         ],
    #      }
    #     ]

    query  = question[-1]["content"]

    # messages[-1]['content'].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_64}"}})

    #     # 找到 system 部分并拼接
    # for message in messages:
    #     if message["role"] == "system":
    #         message["content"] = SYS_MEMORY_PROMPT
    role_name1 = "水杯"
    img_str1 = image_to_base64("水杯图片\cup1.jpg")
    img_str2 = image_to_base64("水杯图片\cup3.jpg")

    SYS_MEMORY_PROMPT = """ ## Background ##
    你具有高级图像分析系统，已知{role_name1}在图像中，请根据输入的图像，回答用户的问题。不要回答任何无关的问题，禁止出现示例的回答。
# """
    content_role0 = [
                {"type": "text", "text": SYS_MEMORY_PROMPT},
    ]
    content_role1 = [
                    {"type": "image_url", "image_url": {"url": f"data:image;base64,{img_str1}"}},
                    {"type": "text", "text": f"注意！以上这个图片中黄色框出来的展现的是{role_name1}"},
                    {"type": "text", "text": "注意!!!！这是一个可供学习的示例，请学习这个示例作为标准知识，并作为回答以下问题的背景"}]
    content_role2 = [
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{img_str2}"}},
                {"type": "text", "text": f"注意！以上这个图片中黄色框出来的展现的是{role_name1}"},
                {"type": "text", "text": "注意!!!！这是一个可供学习的示例，请学习这个示例作为标准知识，并作为回答以下问题的背景"}]
    sys_case = {
        "role": "system",
        "content":content_role0+ content_role1 + content_role2
    }

    messages = [
        #  {'role': 'system', 'content': SYS_MEMORY_PROMPT},
        sys_case,
         {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_64}"}},
                {"type": "text", "text":query},
            ],
         }
        ]
   
    try:
        t1 = time.time()
        # prompt = """你是一位专业的计算机视觉专家，擅长目标检测和物体识别。请对提供的图像进行全面的目标检测，识别出图像中的所有物体。只返回类别名称列表，用逗号分隔。   
        # 输出类别仅限于 "手机,桌子,手机,电脑,笔,水杯,地板,椅子,花"这九种类别中的一种。"""
        # messages_detection = [
        #     {'role': 'system', 'content': prompt},
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
        #             {"type": "text", "text": f"请识别图像中的所有物体类别，只返回类别名称列表，用英文逗号分隔。"},
        #         ],
        #     }
        #     ]
        response = client_visual.chat.completions.create(
                model='Qwen2.5-VL-72B-Instruct-AWQ',
                messages=messages,
                max_tokens = 64,
                stream= True
            )        
        # print(response.choice[0].message.content)
        t2 = time.time()
       

        print(f"视觉大模型请求完成，耗时: {t2 - t1:.2f}秒")

            #非流式返回
            # result = r
            # sponse.json()
            # # 解析返回的JSON，提取类别
            # print(f"✅ 请求成功，状态码: {response.status_code}")
            # decoded_line = response.content.decode("utf-8") 
            # raw_str = json.loads(decoded_line)["response"]["choices"][0]["message"]["content"]
            # # 清洗掉 markdown 的 ```json 和 ``` 标记
            # return raw_str      
            ##流式返回
        return response
            # return Response(generator_head(response,session_id), content_type='text/event-stream')
 
       
    except Exception as e:
        print(f"API调用异常: {e}")
        return "无法确定位置"


# 修改相似度计算函数，使其适用于单帧处理
def compute_frame_similarity(current_frame, last_frame_feature, processor, model, device):
    """
    计算当前帧与上一个保留帧之间的相似度
    
    参数:
        current_frame: 当前帧图像
        last_frame_feature: 上一个保留帧的特征
        processor: CLIP处理器
        model: CLIP模型
        device: 计算设备
    
    返回:
        相似度值，以及当前帧的特征
    """
    # 转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    
    # 归一化特征
    features = features / features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarity = torch.nn.functional.cosine_similarity(
        last_frame_feature, features
    ).item()
    
    return similarity, features.cpu()


class processing_backend:
    def __init__(self, sn: int, stream_url):
        self.sn_str = str(sn)
        self.fresh_time = time.time()
        self.grouding = []
        self.should_stop = False
        self.cached_frame = dict()
        self.request_interval = 2
        # self.processsing_vedio_backend(self.stream_url)
        self.process_video_thread = threading.Thread(target=processing_backend.processsing_vedio_backend,
                                                     args=(self, stream_url), daemon=True)
        self.process_video_thread.start()

    def processsing_vedio_backend(self, stream_url: str):
        """
        使用YOLOv8进行目标检测，并使用CLIP模型进行实时帧压缩
        """
        frame_dir = f"run/{self.sn_str}/frames"
        cls_dir = f"run/{self.sn_str}/cls"
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)
        
        # 获取原始视频的帧率
        cap = cv2.VideoCapture(stream_url)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算需要跳过的帧数，以达到目标帧率1FPS
        frame_skip = max(1, int(original_fps))

        frame_id = 0
        processed_frame_id = 0
        MAX_RETRY = 10
        last_request_time = time.time()
        
        # 加载CLIP模型（预先加载以避免重复加载）
        print("加载CLIP模型...")
        model = CLIPModel.from_pretrained("/data/sjc/models/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("/data/sjc/models/clip-vit-large-patch14")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # 初始化变量
        last_frame = None
        last_frame_feature = None
        similarity_threshold = 0.98
        
        print("开始实时视频处理...")
        while True:
            # 读取帧
            ret, frame = cap.read()

            if self.should_stop:
                print("视频播放结束")
                break
            
            if not ret:
                retry = 0
                while retry < MAX_RETRY:
                    time.sleep(0.5)
                    ret, frame = cap.read()
                    if ret:
                        break
                    retry += 1
                if not ret:
                    print("视频读取失败")
                    break
                
            # 根据时间跳过帧
            current_time = time.time()
            if current_time - last_request_time < self.request_interval:
                frame_id += 1
                continue
            
            # 处理第一帧
            if last_frame is None:
                last_frame = frame.copy()
                # 为第一帧提取特征
                pil_image = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    last_frame_feature = model.get_image_features(**inputs)
                
                # 归一化特征
                last_frame_feature = last_frame_feature / last_frame_feature.norm(dim=-1, keepdim=True)
                
                # 处理第一帧
                image_id = f"img_{processed_frame_id:04d}"
                threading.Thread(target=self.vlm_detetion,
                            args=(last_frame, image_id, frame_dir),
                            daemon=True).start()
                
                print(f"处理第一帧: {image_id}")
                processed_frame_id += 1
            else:
                # 计算当前帧与上一个保留帧的相似度
                similarity, current_feature = compute_frame_similarity(
                    frame, last_frame_feature, processor, model, device
                )
                
                # 如果相似度低于阈值，则处理当前帧
                if similarity < similarity_threshold:
                    image_id = f"img_{processed_frame_id:04d}"
                    threading.Thread(target=self.vlm_detetion,
                                args=(frame.copy(), image_id, frame_dir),
                                daemon=True).start()
                    
                    print(f"处理新帧: {image_id}, 相似度: {similarity:.4f}")
                    
                    # 更新参考帧和特征
                    last_frame = frame.copy()
                    last_frame_feature = current_feature
                    processed_frame_id += 1
                else:
                    print(f"跳过相似帧，相似度: {similarity:.4f}")
            
            last_request_time = current_time
            frame_id += 1
        
        cap.release()
        print("视频处理结束")

    def get_class_to_images(self):
        self.waiting_time = 0
        root_dir = f"run/{self.sn_str}/cls"
        class_to_images = {}
        for filename in os.listdir(root_dir):
            # 拼接完整文件路径
            file_path = os.path.join(root_dir, filename)
            # 确保处理的是文件（而非子目录）
            if os.path.isfile(file_path):
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    # 使用splitlines()自动去除换行符
                    lines = file.read().splitlines()

                # 将结果存入字典
                class_to_images[filename.replace("_to_images.json", "")] = lines
        return class_to_images

    def get_base64_to_id(self):
        return self.cached_frame
        # self.waiting_time = 0
        # while True:
        #     try:
        #         with open(f"run/{self.sn_str}/base64_to_id.json", "r", encoding="utf-8") as f:
        #             datas= f.read().splitlines()
        #             result = dict(map(lambda s: list(json.loads(s).items())[0], datas))

        #             return  result #[json.loads(data) for data in datas]
        #     except Exception as e:
        #         print(f"Error reading base64_to_id.json: {e}")
        #         continue

    def get_grounding(self):
        return self.grouding

    def get_result(self):
        self.fresh_time = time.time()
        return {"class": self.get_class_to_images(), "base64": self.get_base64_to_id()}

    def stop_processing(self):
        self.should_stop = True
        return {'status': 'stopping'}

    def start_processing(self):
        self.should_stop = False
        self.fresh_time=time.time()
        return {'status': 'processing'}

    def vlm_detetion(self, frame, frame_id, frame_dir):
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        role_name1 = "水杯"
        img_str1 = image_to_base64("prompt_picture/cup2.jpg")
        img_str2 = image_to_base64("prompt_picture/cup3.jpg")

        SYS_MEMORY_PROMPT = """ 你是一位专业的计算机视觉专家，擅长目标检测和物体识别。请对提供的图像进行全面的目标检测，识别出图像中的所有物体。只返回类别名称列表，用逗号分隔。   
        输出类别仅限于 "手机,桌子,手机,电脑,笔,水杯,地板,椅子,花"这九种类别中的一种,不允许出现不在这九种的类别。"""

        content_role0 = [
                    {"type": "text", "text": SYS_MEMORY_PROMPT},
        ]
        content_role1 = [
                        {"type": "image_url", "image_url": {"url": f"data:image;base64,{img_str1}"}},
                        {"type": "text", "text": f"注意！以上这个图片中黄色框出来的展现的是{role_name1}"},
                        {"type": "text", "text": "注意!!!！这是一个可供学习的示例，请学习这个示例作为标准知识，并作为回答以下问题的背景"}]
        content_role2 = [
                    {"type": "image_url", "image_url": {"url": f"data:image;base64,{img_str2}"}},
                    {"type": "text", "text": f"注意！以上这个图片中黄色框出来的展现的是{role_name1}"},
                    {"type": "text", "text": "注意!!!！这是一个可供学习的示例，请学习这个示例作为标准知识，并作为回答以下问题的背景"}]
        sys_case = {
            "role": "system",
            "content":content_role0+ content_role1 + content_role2
        }
        messages_detection = [
            #  {'role': 'system', 'content': SYS_MEMORY_PROMPT},
            sys_case,
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text":f"请识别图像中的所有物体类别，只返回类别名称列表，用英文逗号分隔。"},
                ],
            }
            ]
        try:
            t1 = time.time()
            response = client_visual.chat.completions.create(
                model='Qwen2.5-VL-72B-Instruct-AWQ',
                messages=messages_detection,
                max_tokens=64,
                timeout = 5
            )
            t2 = time.time()
            print(f"大模型处理{frame_id},用时{t2 - t1}")
            # 解析为字典
            raw_data = response.choices[0].message.content.strip("`").strip("json").strip()
            results = raw_data.split(",")
            # payload = {
            #     "stream": False,
            #     "messages": messages_detection,
            #     "max_tokens": 1024
            # }

            # headers = {"Content-Type": "application/json"}
            # t1 = time.time()

            # response = requests.post(URL_72B, json=payload, headers=headers, timeout=100, stream=False)
            # t2 = time.time()
            # print(f"大模型请求耗时:{t2-t1}")
            # decoded_line = response.content.decode("utf-8")

            # raw_str = json.loads( decoded_line)["response"]["choices"][0]["message"]["content"]
            # # 解析为数组
            # results = raw_str.split(",")

            categories = []
            for cls_name in results:
                categories.append(cls_name)

            # 保存帧到本地frame文件夹
            frame_path = os.path.join(frame_dir, f"{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"保存帧到: {frame_path}")

            # 只有当检测到类别时才保存图像
            if categories and categories[0]:
                # 将图像转换为base64
                # resized_frame = np.resize(frame, (720, 1280, 3))
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                for cls in categories:
                    while True:
                        try:
                            with open(f"run/{self.sn_str}/cls/{cls}_to_images.json", "a", encoding="utf-8") as f:
                                f.write(str(frame_id) + '\n')
                                break
                        except:
                            print("写入class_to_images文件错误")

                self.cached_frame[frame_id] = image_base64

                print(f"[{frame_id}] 识别到类别: {categories}")
            else:
                print(f"[{frame_id}] 未检测到任何类别，跳过保存")
        except Exception as e:
            ##备用模型
            print(e)


def process_question(class_to_images: Dict[str, List[str]], base64_to_id: Dict[str, str], message: str,
                     session_id: int):
    print("\nmessage：", message)
    # 获取所有可用的类别
    available_categories = list(class_to_images.keys())
    categories = extract_categories_from_question(available_categories, message)
    print("识别出的问题相关类别：", categories)

    # 收集所有相关类别的图片ID
    all_image_ids = set()
    for category in categories:
        if category in class_to_images:
            all_image_ids.update(class_to_images[category])

    # 去重后的图片ID列表
    unique_image_ids = list(all_image_ids)
    print(f"找到 {len(unique_image_ids)} 张相关图片")

    if unique_image_ids:
        # 获取这些图像ID对应的base64编码
        unique_image_ids = unique_image_ids[-2:]
        images_base64 = [base64_to_id[img_id] for img_id in unique_image_ids]
        # 将所有相关图片一起发送给模型
        return ask_model_about_category(images_base64, message, categories)
    else:
        print("\n未找到相关图片，无法回答问题。")
        return "未找到相关图片，无法回答问题。"


@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    print(data)

    stream_url = data.get('stream_url')
    flag = data.get('flag')
    sn = data.get('sn')

    def generate():
        # Check if stream is already being processed
        if sn in processing_pool:
            backend = processing_pool[sn]
        else:
            # Start processing the video
            backend = processing_backend(sn, stream_url)
            processing_pool[sn] = backend
        while True:
            # Check if processing has stopped
            if backend.should_stop:
                yield f"data: {json.dumps({'status': 'stopped'})}\n\n"
                break
            # Get latest results
            results = backend.get_grounding()

            # print(results)
            yield f"data: {json.dumps(results, cls=NumpyEncoder, ensure_ascii=False)}\n\n"
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

    def start():
        return Response(generate(), content_type='text/event-stream')  # Newline Delimited JSON
        # yield jsonify({'result': processing_pool.get_grounding()})

    def stop(sn):
        if sn in processing_pool:
            backend = processing_pool[sn]
            result = backend.stop_processing()
            time.sleep(0.1)
            return jsonify(result)
        else:
            return jsonify({'status': 'error', 'message': 'Stream not found'})

    if flag == "start":
        return start()
    if flag == "stop":
        return stop(sn)
    return jsonify({'status': 'error', 'message': '没有标志位'})


@app.route('/memory_vqa', methods=['POST'])
def memory_vqa():
    data = request.json
    sn = data.get('sn')

    message = data.get('message')
    session_id = data.get('session id')
    logging.info(data)

    if not sn or not message:
        return jsonify({'error': 'Missing sn or message parameter'})

    local_llm = QwenLLM()

    visual_llm = VisualLLM(
        sn=sn
    )
    output_llm = SearchLLM(session_id, local_llm, visual_llm)
    query = message[-1]["content"]
    messages_local = [
        {'role': 'system', 'content': """你是一个专业的视觉问答助手。你的任务是生成一个简短的垫词，为后续的视觉分析结果做铺垫。
                    垫词应该简洁明了，直接点明用户的问题，并引导用户关注即将到来的视觉分析结果。
                    不要解释你的分析过程，直接给出垫词即可，不超过10个字，。"""},
        {'role': 'user', 'content': query}
    ]

    return Response(output_llm.chat_stream(session_id, messages_local, message), content_type='text/event-stream')


def thread_manage():
    while True:
        time.sleep(1)
        for k, v in processing_pool.items():
            time.sleep(1)
            if isinstance(v, processing_backend):
                print(f"sn{v.sn_str}:last process time:{time.time() -v.fresh_time}")
                if time.time() - v.fresh_time > MAX_PROCESSING_TIMEOUT:
                    print(f"\n\n\n\n\nsn{v.sn_str}:stopped after {time.time() - v.fresh_time}\n\n\n\n\n\n")
                    v.should_stop = True
                    time.sleep(1)
                    processing_pool.pop(k)
                    # del v
                    break


if __name__ == "__main__":
    check_thread = threading.Thread(target=thread_manage, daemon=True)
    check_thread.start()
    app.run(host='0.0.0.0', port=9005, threaded=True)
    
    # sn = 1
    # backend1 = processing_backend(sn, 0)
    # processing_pool[sn]=backend1
    # sn+=1
    # backend2 = processing_backend(sn, 0)
    # processing_pool[sn]=backend2
    # sn+=1
    # backend3 = processing_backend(sn, 0)
    # processing_pool[sn]=backend3
    # sn+=1
    # backend4 = processing_backend(sn, 0)
    # processing_pool[sn]=backend4
    # sn+=1
    # backend5 = processing_backend(sn, 0)
    # processing_pool[sn]=backend5
    # sn+=1
    # backend6 = processing_backend(sn, 0)
    # processing_pool[sn]=backend6
    # sn+=1
    # backend7 = processing_backend(sn, 0)
    # processing_pool[sn]=backend7
    # sn+=1
    # backend8 = processing_backend(sn, 0)
    # processing_pool[sn]=backend8
    # sn+=1
    # backend9 = processing_backend(sn, 0)
    # processing_pool[sn]=backend9
    # while len(threading.enumerate())>1:
    #     time.sleep(0.5)
    #     print(threading.enumerate())
    #     continue
    
