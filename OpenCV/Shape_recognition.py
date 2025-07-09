import cv2
import numpy as np

# 윤곽선에서 도형 종류를 판단하는 함수
def detect_shape(contour):
    shape = "Unidentified"  # 기본값

    # 외곽선 길이 계산 (둘레)
    peri = cv2.arcLength(contour, True)

    # 윤곽선을 단순화하여 꼭짓점 수를 줄임 (정밀도: 둘레의 4%)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    # 꼭짓점 개수로 도형 판단
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"  # 삼각형
    elif vertices == 4:
        # 4개의 꼭짓점이면 정사각형 또는 직사각형
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)  # 가로세로 비율 계산
        shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    elif vertices == 5:
        shape = "Pentagon"  # 오각형
    elif vertices > 5:
        shape = "Circle"  # 원으로 간주
    return shape

# 웹캠 열기 (기본 카메라: index 0)
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break  # 카메라 연결 실패 시 종료

    # 전처리 단계
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        # 흑백 변환
    blur = cv2.GaussianBlur(gray, (5, 5), 1)              # 노이즈 제거를 위한 블러 처리
    edged = cv2.Canny(blur, 50, 150)                      # 엣지(윤곽선) 검출

    # 외곽선 찾기 (RETR_EXTERNAL: 외곽선만, CHAIN_APPROX_SIMPLE: 꼭 필요한 점만 저장)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # 너무 작은 도형은 무시
            shape = detect_shape(cnt)                     # 도형 인식
            x, y, w, h = cv2.boundingRect(cnt)            # 외곽 사각형 추출
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)      # 윤곽선 그리기
            cv2.putText(frame, shape, (x, y - 10),         # 텍스트로 도형 이름 출력
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("Shape Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
