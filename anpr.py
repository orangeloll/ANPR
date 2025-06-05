import numpy as np
import pytesseract
import cv2
import re
import tkinter as tk
from tkinter import filedialog
import os # os 모듈 추가
from PIL import ImageFont, ImageDraw, Image # Pillow 라이브러리 임포트

# Tesseract 경로 지정 (기존과 동일)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def classify_plate(text):
    match = re.match(r'(\d{2,3})([가-힣])(\d{4})', text)
    if not match:
        return "형식 오류", "형식 오류", "없음"

    num, letter, serial = match.groups()
    num = int(num)

    # 차량 종류
    if 1 <= num <= 99:
        if num <= 69:
            car_type = "승용차"
        elif num <= 79:
            car_type = "승합차"
        elif num <= 97:
            car_type = "화물차"
        else:
            car_type = "특수차"
    elif 100 <= num <= 999:
        if num <= 699:
            car_type = "승용차"
        elif num <= 799:
            car_type = "승합차"
        elif num <= 979:
            car_type = "화물차"
        elif num <= 997:
            car_type = "특수차"
        else:
            car_type = "긴급차"
    else:
        car_type = "잘못된 범위입니다" # 기본값 (범위 외일 때)

    # 용도
    if letter in ['허', '하', '호']:
        usage = "렌터카"
    elif letter == '배':
        usage = "택배 차량"
    elif letter in ['아', '바', '사', '자']:
        usage = "운수업 (버스, 택시 등)"
    else:
        usage = "자가용"

    return car_type, usage, serial


# --- 파일 선택 및 이미지 로드 부분 수정 ---
root = tk.Tk()
root.withdraw() # Tkinter 창(빨간 네모 부분 창)을 숨김

file_path = filedialog.askopenfilename(
    title="이미지 파일 선택",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("이미지 파일을 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# cv2.imread() 대신 cv2.imdecode() 사용 -> imread 함수가 한글이나 특정 특수 문자를 인식하지 못해서 변경
# 파일을 바이너리 모드로 읽어 numpy 배열로 변환 후 디코딩
try:
    with open(file_path, 'rb') as f:
        # 파일 내용을 바이트 스트림으로 읽어들입니다.
        img_bytes = np.frombuffer(f.read(), np.uint8)
    # imdecode를 사용하여 이미지 데이터를 디코딩합니다.
    img_ori = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
except Exception as e:
    raise Exception(f"이미지 파일을 읽거나 디코딩하는 중 오류가 발생했습니다: {e}")


if img_ori is None:
    # imdecode가 None을 반환하는 경우는 보통 파일이 유효한 이미지 형식이 아니거나 손상된 경우
    raise Exception("선택한 이미지를 로드할 수 없습니다. 파일이 유효한 이미지 파일인지 확인해주세요.")


height, width, channel = img_ori.shape


MAX_DIM = 800 # 최대 가로/세로 800 픽셀로 제한
if height > MAX_DIM or width > MAX_DIM:
    print(f"이미지 크기가 너무 큽니다 ({width}x{height}). 리사이즈합니다.")
    scale = MAX_DIM / max(height, width)
    img_ori = cv2.resize(img_ori, (int(width * scale), int(height * scale)))
    height, width, channel = img_ori.shape
    print(f"새로운 크기: {width}x{height}")

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=11
)
contours, _ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)

    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

MIN_AREA = 90
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = [] ##가능성이 있는 곳

cnt = 0
for d in contours_dict:
    area = d['w']*d['h']
    ratio = d['w']/d['h']

    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)
##vis = img_ori.copy()
temp_result = np.zeros((height, width, channel), dtype=np.uint8)
for d in possible_contours:
    ##cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

MAX_DIAG_MULTIPLYER = 5 ##첫번째 컨투어와 다음 사이의 대각선 5배 안에 있어야함
MAX_ANGLE_DIFF = 12.0 ##중심을 이었을때 벌어진 정도
MAX_AREA_DIFF = 0.5 ##면적 차이
MAX_WIDTH_DIFF = 0.8 ##너비차이
MAX_HEIGHT_DIFF = 0.2 ##높이
MIN_N_MATCHED = 6 ## 위 조건들을 만족하는 애들이 이 개수 미만이면 후보가 아님

def find_chars(contour_list): ##나중에 재귀함수로 계속 찾기때문에 지정
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']: ##같은 인덱스 값을 가진 컨투어끼린 비교할 필요가 없음 > 같은 애인거니까
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w']**2+d1['h']**2)

            ##벡터 a와 벡터 b 사이의 거리를 구한다 {np.linalg.norm(a-b)}
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            
            angle_diff = np.degrees(np.arctan2(dy, dx))
            ## tanθ = dy/dx, θ = arctan dy/dx, np.arctan() > 아크탄젠트 값을 구한다(라디안) > np.degrees() >라디안을 도로 변경한다
            area_diff = abs(d1['w']*d1['h'] - d2['w']*d2['h'])/(d1['w']*d1['h'])
            width_diff = abs(d1['w']-d2['w'])/d1['w']
            height_diff = abs(d1['h']-d2['h'])/d1['h']

            if distance < diagonal_length1*MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED: ##컨투어가 6개보다 작으면 번호판일 수가 X 한국 번호판은 6자리가 넘기때문에
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])
        ## np.take(a, idx) > a에서 idx와 같은 인덱스의 값만 추출
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # recursive
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx

result_idx = find_chars(possible_contours)


matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))


temp_result = np.zeros((height, width, channel), dtype=np.uint8)
##vis = img_ori.copy()
for r in matched_result:
    for d in r:
#       cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        ##cv2.rectangle(vis, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 0), thickness=2)

##cv2.imshow("img1",vis)

PLATE_WIDTH_PADDING = 1.31 # 1.3에서 수정해보기
PLATE_HEIGHT_PADDING = 1.51 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )

    plate_ratio = img_cropped.shape[1] / img_cropped.shape[0]
    if plate_ratio < MIN_PLATE_RATIO or plate_ratio > MAX_PLATE_RATIO:
        continue


    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
longest_idx, longest_text = -1, 0
plate_chars = []


for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h

    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    # 윤곽선 강화
    ##kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ##img_result = cv2.dilate(img_result, kernel, iterations=1)

    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 1') ##psm와 oem 값에 따라 특정 사진의 인식 정확도가 달라지기도 함
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c

    if '파' in result_chars:
        result_chars = result_chars.replace('파', '마')
        print(f"DEBUG: '파'를 '마'로 교정했습니다. 교정 후: '{result_chars}'")
    elif '지' in result_chars:
        result_chars = result_chars.replace('지', '저')
    elif '리' in result_chars:
        result_chars = result_chars.replace('리', '러')

    # 첫 글자가 숫자가 아니면 제거
    if result_chars and not result_chars[0].isdigit():
        result_chars = result_chars[1:]
    

    match = re.match(r'^(\d{2,3})[가-힣]', result_chars)
    

    if match:
        num_part = match.group(1)
        if len(num_part) == 2 and len(result_chars) > 7: ## 앞자리가 2자리일때 총 길이가 8이상이면 뒤를 자름
            result_chars = result_chars[:7]
        elif len(num_part) == 3 and len(result_chars) > 8: ## 앞자리가 3자리일때 총 길이가 9이상이면 뒤를 자름
            result_chars = result_chars[:8]
        
    else:
        print(f"❌ 번호판 형식 인식 실패: {result_chars}")
        continue # 이 루프(i) 건너뜀

    print("번호판:", result_chars)
    car_type, usage, serial = classify_plate(result_chars)

    print(f"▶ 차량 종류: {car_type}, 용도: {usage}, 등록번호: {serial}")
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

#cv2.imshow("번호판 창", img_result) # 인식된 번호판 영역을 보여주는 창
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# --- Pillow를 사용하여 한글 텍스트 그리기 ---
if longest_idx != -1:
    x_ori, y_ori, w_ori, h_ori = plate_infos[longest_idx]['x'], plate_infos[longest_idx]['y'], plate_infos[longest_idx]['w'], plate_infos[longest_idx]['h']

    # OpenCV 이미지를 Pillow 이미지로 변환 (BGR -> RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 폰트 로드 (시스템에 설치된 한글 폰트 경로 지정)
    font_path = "C:/Windows/Fonts/malgunbd.ttf" # 맑은 고딕 볼드


    try:
        font = ImageFont.truetype(font_path, 25) # 폰트 크기 조정
    except IOError:
        print(f"경고: '{font_path}' 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    # 번호판 외곽선 그리기
    draw.rectangle([(x_ori, y_ori), (x_ori + w_ori, y_ori + h_ori)], outline=(0, 255, 0), width=2)

    # 번호판 텍스트 그리기
    text_to_display = plate_chars[longest_idx]
    draw.text((x_ori, y_ori - 35), text_to_display, font=font, fill=(0, 255, 0)) # 텍스트 위치 조정 (y_ori - 35)

    # 차량 정보 텍스트 그리기
    car_type, usage, serial = classify_plate(text_to_display)
    
    # 텍스트 줄 간격 계산 (대략적인 폰트 높이 + 여백)
    # font.getbbox("가")는 텍스트 바운딩 박스를 반환합니다. (left, top, right, bottom)
    # bottom - top 으로 대략적인 높이를 구할 수 있음
    try:
        line_height = font.getbbox("가")[3] - font.getbbox("가")[1] + 5
    except AttributeError: # 이전 Pillow 버전에서는 getsize를 사용
        line_height = font.getsize("가")[1] + 5


    info_x = x_ori
    info_y = y_ori + h_ori + 10 # 번호판 아래 10px 간격

    draw.text((info_x, info_y), f"종류: {car_type}", font=font, fill=(255, 0, 0)) # 종류는 빨간색
    draw.text((info_x, info_y + line_height), f"용도: {usage}", font=font, fill=(255, 0, 0)) # 용도는 빨간색, 종류 아래 5px 간격

    # Pillow 이미지를 다시 OpenCV 이미지로 변환 (RGB -> BGR)
    img_ori = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cv2.imshow("Original Image with Detected Plate and Info", img_ori)
if longest_idx != -1:
    cv2.imshow("Detected Plate Region for OCR", plate_imgs[longest_idx])
cv2.waitKey(0)
cv2.destroyAllWindows()