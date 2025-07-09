import cv2
import numpy as np

# === 웹캠 열기 ===
cap = cv2.VideoCapture(0)  # 0번 카메라 사용

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()  # 카메라가 열리지 않으면 종료

# === 실시간 영상 처리 루프 ===
while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 프레임을 못 읽으면 종료

    # BGR → HSV 색공간으로 변환
    # OpenCV에서 기본은 BGR이며, 색상 필터링에는 HSV가 더 적합
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # === 빨간색 HSV 범위 설정 ===
    # 빨간색은 HSV 색상값 H가 0도와 180도 양쪽 끝에 걸쳐 있음
    # 그래서 2개의 범위로 나누어 탐지해야 함

    # 첫 번째 빨간색 범위 (약 0° ~ 10°)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    # 두 번째 빨간색 범위 (약 170° ~ 180°)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 각 범위에 해당하는 픽셀만 흰색(255)으로 표시한 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 두 마스크를 합쳐서 빨간색 전체 범위 마스크 생성
    red_mask = mask1 | mask2

    # 빨간색 영역만 추출: 마스크를 원본 프레임에 적용
    red_detected = cv2.bitwise_and(frame, frame, mask=red_mask)

    # === 영상 출력 ===
    cv2.imshow("Original", frame)            # 원본 영상
    cv2.imshow("Red Mask", red_detected)     # 빨간색만 추출된 영상

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 자원 해제 및 종료 ===
cap.release()               # 카메라 장치 해제
cv2.destroyAllWindows()     # 모든 창 닫기
