import cv2
from ultralytics import YOLO  # Ultralytics YOLOv5su 라이브러리 불러오기

# === 1. 사전학습된 YOLOv8 모델 로드 ===
# coco 데이터셋으로 학습된 'yolov8n.pt' 모델 로드 (가장 가볍고 빠름)
model = YOLO("yolov5su.pt")  # yolov8s.pt, yolov8m.pt 등 다른 모델도 가능

# === 2. 카메라 열기 ===
cap = cv2.VideoCapture(0)  # 0번은 기본 내장 웹캠

if not cap.isOpened():  # 웹캠 연결 실패 시
    print("카메라 열기 실패")
    exit()

# === 3. 실시간 프레임 처리 루프 ===
while True:
    ret, frame = cap.read()  # 프레임 읽기 (ret: 성공 여부, frame: 이미지)
    if not ret:
        break  # 카메라가 프레임을 못 가져오면 종료

    # === 4. YOLOv8 모델을 이용한 객체 인식 ===
    results = model(frame, stream=True)  
    # stream=True: 내부적으로 메모리를 절약하며 여러 결과를 반복(iterable)로 리턴

    # === 5. 결과에서 탐지된 객체(boxes)들을 하나씩 처리 ===
    for r in results:              # 결과 묶음 하나씩 반복 (이미지는 1장이므로 1번만 반복)
        for box in r.boxes:        # 탐지된 객체 수만큼 반복
            # box.xyxy[0]: (x1, y1, x2, y2) 좌표 가져오기 (float → int 변환)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # box.conf[0]: 객체 탐지 신뢰도 confidence (0.0 ~ 1.0)
            conf = float(box.conf[0])

            # box.cls[0]: 클래스 인덱스 (예: 0은 'person', 1은 'bicycle' 등)
            cls = int(box.cls[0])

            # model.names[cls]: 실제 클래스 이름 가져오기
            label = model.names[cls]

            # === 6. 바운딩 박스 및 클래스 이름을 영상에 표시 ===
            # 사각형 그리기 (초록색, 두께 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 텍스트 라벨 출력 (예: "person 0.85")
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === 7. 결과 영상 출력 ===
    cv2.imshow("YOLOv8 Detection", frame)  # 인식된 객체와 라벨이 그려진 프레임 표시

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 8. 종료 처리 ===
cap.release()             # 웹캠 해제
cv2.destroyAllWindows()   # 모든 창 닫기