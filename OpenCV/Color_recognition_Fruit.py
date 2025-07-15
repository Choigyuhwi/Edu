import cv2
import numpy as np

# HSV 색상 범위 정의: (이름, HSV 최솟값, HSV 최댓값, 출력할 BGR 색)
colors = [
    ("red",    (0, 100, 100),   (10, 255, 255),   (0, 0, 255)),       # 빨간색
    ("orange", (11, 100, 100),  (20, 255, 255),   (0, 128, 255)),     # 주황색
    ("yellow", (15, 80, 80),    (40, 255, 255),   (0, 255, 255)),     # 노란색
    ("green",  (45, 100, 100),  (75, 255, 255),   (0, 255, 0)),       # 초록색
    ("sky",    (76, 100, 100),  (95, 255, 255),   (255, 255, 0)),     # 하늘색
    ("blue",   (100, 100, 100), (130, 255, 255),  (255, 0, 0)),       # 파란색
    ("pupple", (131, 100, 100), (160, 255, 255),  (255, 0, 255)),     # 보라색
    ("pink",   (161, 100, 100), (170, 255, 255),  (255, 128, 255)),   # 분홍색
    ("brown",  (10, 150, 20),   (20, 200, 200),   (19, 69, 139)),     # 갈색 (임의 BGR)
    ("black",  (0, 0, 0),       (180, 255, 50),   (0, 0, 0))          # 검정색
]

# 카메라 장치 열기 (0번 카메라 사용)
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 프레임을 못 읽으면 종료

    # BGR 이미지를 HSV 색공간으로 변환 (색상 인식을 위해 필수)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 정의된 각 색상 범위마다 반복
    for name, lower, upper, bgr in colors:
        # 현재 색상에 해당하는 마스크 이미지 생성
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # 마스크에서 외곽선 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 검출된 모든 윤곽선에 대해 처리
        for cnt in contours:
            area = cv2.contourArea(cnt)  # 윤곽선 면적 계산
            if area > 500:  # 너무 작은 Q무시
                x, y, w, h = cv2.boundingRect(cnt)  # 외곽 사각형 좌표 계산
                cx, cy = x + w // 2, y + h // 2      # 중심 좌표 계산 (선택적 활용 가능)

                # 인식된 영역에 사각형 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)

                # 인식된 색상의 이름 텍스트 표시
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    # 결과 프레임을 화면에 출력
    cv2.imshow("Color Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 자원 해제
cap.release()
cv2.destroyAllWindows()
