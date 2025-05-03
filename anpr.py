import numpy as np, pytesseract, cv2
import re

#Tesseract 경로 지정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def classify_plate(text):
    match = re.match(r'(\d{2,3})([가-힣])(\d{4})', text)
    if not match:
        return "형식 오류", "형식 오류", "없음"

    num, letter, serial = match.groups()
    num = int(num)

    # 차량 종류
    if 100 <= num <= 699:
        car_type = "승용차"
    elif 700 <= num <= 799:
        car_type = "승합차"
    elif 800 <= num <= 979:
        car_type = "화물차"
    elif 980 <= num <= 997:
        car_type = "특수차"
    elif 998 <= num <= 999:
        car_type = "긴급차"
    else:
        car_type = "승용차"

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
img_ori = cv2.imread("images/car10.jpg")

height, width, channel = img_ori.shape

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

contours, _ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8) ##contour 확인

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



MIN_AREA = 80
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
vis = img_ori.copy()
temp_result = np.zeros((height, width, channel), dtype=np.uint8)
for d in possible_contours:
#     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
    
MAX_DIAG_MULTIPLYER = 5 ##첫번째 컨투어와 다음 사이의 대각선 5배 안에 있어야함
MAX_ANGLE_DIFF = 12.0 ##중심을 이었을때 벌어진 정도
MAX_AREA_DIFF = 0.5 ##면적 차이
MAX_WIDTH_DIFF = 0.8 ##너비차이
MAX_HEIGHT_DIFF = 0.2 ##높이
MIN_N_MATCHED = 3 ## 위 조건들을 만족하는 애들이 3개 미만이면 후보가 아님

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
            ##dx가 0값이면 믐 같은 위치임 그럼 번호판일 확률이 적고 dx가 0이면 arctan dy/dx의 분모가 0이라 나눌 수 없으니 오류나니까 처리해주는 거
            if dx == 0:
                angle_diff = 90
            else:
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

        if len(matched_contours_idx) < MIN_N_MATCHED: ##컨투어가 3개보다 작아! 그럼 번호판일 수가.. 한국 번호판은 3자리넘으니까
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

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)
vis = img_ori.copy()
for r in matched_result:
    for d in r:
#         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        cv2.rectangle(vis, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 0), thickness=2)
        
cv2.imshow("vis", vis)

PLATE_WIDTH_PADDING = 1.3 # 1.3에서 수정해보기
PLATE_HEIGHT_PADDING = 1.5 # 1.5
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
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
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
    
    # find contours again (same as above)
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
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 8 --oem 1')
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print("번호판:",result_chars)
    car_type, usage, serial = classify_plate(result_chars)
    print(f"▶ 차량 종류: {car_type}, 용도: {usage}, 등록번호: {serial}")
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i



cv2.imshow("image", img_result)
cv2.waitKey(0)
