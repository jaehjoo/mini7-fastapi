from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel
import openai
import os
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from haversine import haversine

app = FastAPI()
path = os.getcwd()

class Request(BaseModel):
    input_txt: str
    hospital_num: int
    start_lat: float
    start_lng: float

@app.get('/api/hello')
async def hello():
  return JSONResponse(content={'message': 'hello!'})

@app.get('/api/order')
async def check_order(req: Annotated[Request, Query()]):
    hospital_df_path = os.path.join(path, "emer_hospital_info.csv")
    ai_model_path = os.path.join(path, "fine_tuned_bert")
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    c_id = os.getenv('MAP_ID')
    c_key = os.getenv('MAP_KEY')
    emergency = pd.read_csv(hospital_df_path)

    # 모델, 토크나이저 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(ai_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ai_model_path)

    # 2. 데이터 처리(pipeline) ---------------------------------
    # 처리
    text = text_summary(req.input_txt)
    predicted_class, _ = predict(text, model, device, tokenizer)

    # ---------------------------------

    if predicted_class <= 1 :
        result = recommendation(emergency, req.start_lat, req.start_lng, req.hospital_num, c_id, c_key)
        if result is None:
            return {
                'message': '경위도 재설정 필요.'
            }
        response = {"em_class": predicted_class + 1, "hospital": result.to_dict(orient="records")}
        return JSONResponse(content=response)
    else:
        print("개인 건강관리")
        return {
            'message': '응급 조치가 불필요한 환자입니다.'
        }

# audio2text
def audio_to_text(client, audio_path, filename):
  # 오디오 파일을 읽어서, 위스퍼를 사용한 변환
  audio_file = open(audio_path+filename, "rb")
  transcription = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file,
      language="ko",
      response_format='text',
  )

  # 결과 반환
  return transcription

# text2summary
def text_summary(input_text):
  # OpenAI 클라이언트 생성
  client = openai.OpenAI()

  # 시스템 역할과 응답 형식 지정
  system_role = '''당신은 훌륭한 응급구조대원입니다. 당신은 환자의 어눌한 말을 듣고 핵심 내용 위주로 정보를 요약하여 답을 줍니다.
  응답은 다음의 형식을 지켜주세요.
  {"summary": \"텍스트 요약\",
  "keyword" : \"핵심 키워드(3가지)\"}'''

  # 입력데이터를 GPT-3.5-turbo에 전달하고 답변 받아오기
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {
              "role": "system",
              "content": system_role
          },
          {
              "role": "user",
              "content": input_text
          }
      ]
  )
  # 응답 받기
  answer = response.choices[0].message.content
  parsed_answer = json.loads(answer)
  summary = parsed_answer['summary']
  keyword = parsed_answer['keyword']

  # 응답형식을 정리하고 return
  return summary, keyword

def text2summary(audio_df):
  summary_lst = []
  keyword_lst = []
  for raw in audio_df['text']:
    summary, keyword = text_summary(raw)
    summary_lst.append(summary)
    keyword_lst.append(keyword)
  audio_df['summary'] = summary_lst
  audio_df['keyword'] = keyword_lst
  return audio_df

# predict
def predict(text, model, device, tokenizer):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities[0], dim=-1).item()
    return pred, probabilities[0]

def get_distance(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"  # 실시간 빠른 길 옵션
    }

    # 요청하고, 답변 받아오기
    response = requests.get(url, headers=headers, params=params).text
    response = json.loads(response)

    dist = int(response['route']['trafast'][0]['summary']['duration'] * (10 ** -3) * (60 ** -1))  # min(분)

    return dist

# recommendation
def recommendation(hospital_df, s_lat, s_lng, hospital_num, c_id, c_key):
  if s_lat < 34 or s_lat > 39 or s_lng < 126 or s_lng > 130:
    print("경위도가 범위를 넘어섰습니다. 값을 다시 한 번 확인해주세요")
    return None
  for step in np.arange(0.1, 3, 0.1):
    distance = set()
    hospital_range_df = hospital_df.loc[(hospital_df['위도'].between(s_lat-step, s_lat+step)) & (hospital_df['경도'].between(s_lng-step, s_lng+step))]
    for i in hospital_range_df.index:
      distance.add((i, haversine((s_lat, s_lng), (hospital_range_df.loc[i, '위도'], hospital_range_df.loc[i, '경도']), unit='km')))
    if len(distance) >= hospital_num:
      distance = sorted(distance, key=lambda x:x[1])[:hospital_num]
      break
  distance = distance if len(distance) == hospital_num else sorted(distance, key=lambda x:x[1])
  dist = []
  for i in distance:
    d_lat = hospital_df.loc[i[0], '위도']
    d_lon = hospital_df.loc[i[0], '경도']
    dist.append((i, get_distance(s_lat, s_lng, d_lat, d_lon, c_id, c_key)))
  fast_dist = sorted(dist, key=lambda x:x[1])[:hospital_num]
  fast_hospital = hospital_df.loc[[item[0][0] for item in fast_dist]].copy()
  fast_hospital['소요시간(분)'] = [item[1] for item in fast_dist]
  fast_hospital.reset_index(inplace=True)
  return fast_hospital