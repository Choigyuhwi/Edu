import cv2

# 카메라 열기 (기본 웹캠은 보통 0번)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 한 프레임 읽기
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow('Camera', frame)  # 창에 영상 출력

    # q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()