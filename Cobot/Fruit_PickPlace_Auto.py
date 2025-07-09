# -------------------------
# 📌 라이브러리 불러오기
# -------------------------

import sys  # 시스템 종료 및 인수 처리를 위한 표준 라이브러리
import cv2  # OpenCV - 컴퓨터 비전 라이브러리
import numpy as np  # NumPy - 행렬 및 수치 계산용
import math  # 삼각함수 및 수학 계산용
import time  # 시간 지연 및 시간 측정용
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox  # PyQt5 GUI 컴포넌트
from PyQt5.QtCore import QTimer  # 타이머(반복 작업)용
from pymycobot.mycobot320 import MyCobot320  # MyCobot 320 로봇 제어용 클래스

# -------------------------
# 📌 메인 윈도우 클래스 정의
# -------------------------
class MyCobotPickupApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 🎥 카메라 캡처 객체 생성 (기본 장치 0번 사용)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ 카메라를 열 수 없습니다.")  # 연결 실패 시 오류 발생

        # 🤖 MyCobot 로봇 제어 객체 생성 (포트 번호와 보레이트 지정)
        self.mycobot = MyCobot320("COM11", 115200)

        # 📌 인식 및 좌표 관련 변수 초기화
        self.roi_coords = None           # ROI 영역 (x1, y1, x2, y2)
        self.roi_marker_pts = None       # ROI 마커 중심 좌표 2개 (회전각 계산용)
        self.latest_coords = None        # 실시간 객체 중심 좌표
        self.locked_coords = None        # 이동 시점에 고정된 객체 중심 좌표
        self.update_enabled = True       # 객체 중심 좌표를 실시간 업데이트할지 여부

        # 🪟 윈도우 UI 초기화
        self.setWindowTitle("HSV 기반 객체 중심 인식 및 이동")  # 창 제목 설정
        self.setGeometry(100, 100, 600, 250)  # 위치(x, y), 크기(width, height)

        # ✅ 상태 표시 라벨 생성 (ROI나 객체 인식 상태, 작업 결과 등을 표시)
        self.status_label = QLabel("ROI 인식 대기 중", self)
        self.status_label.setGeometry(30, 180, 500, 30)

        # -------------------------
        # 📌 버튼 및 UI 위젯 구성
        # -------------------------

        # 🏠 홈 위치로 로봇 이동 버튼
        self.home_btn = QPushButton("🏠 홈 위치 이동", self)
        self.home_btn.setGeometry(30, 30, 150, 40)
        self.home_btn.clicked.connect(self.go_home_position)  # 클릭 시 동작할 함수 연결

        # 🎯 객체 위로 이동 (Z축 높이 유지하며 객체 상단으로 이동)
        self.move_btn = QPushButton("🎯 객체 위로 이동", self)
        self.move_btn.setGeometry(200, 30, 150, 40)
        self.move_btn.clicked.connect(self.move_above_object)

        # 📦 픽업 버튼 (Z축을 내려서 집기 동작 수행)
        self.pickup_btn = QPushButton("📦 픽업", self)
        self.pickup_btn.setGeometry(370, 30, 150, 40)
        self.pickup_btn.clicked.connect(self.pickup_object)

        # 📍 A/B/C/D 위치 선택 콤보박스 (드롭다운)
        self.place_combo = QComboBox(self)
        self.place_combo.setGeometry(30, 90, 150, 40)
        self.place_combo.addItems(["A", "B", "C", "D"])  # 플레이스 위치 선택 가능

        # 🚚 플레이스 버튼 (선택된 위치로 이동 후 물체 놓기)
        self.place_btn = QPushButton("플레이스", self)
        self.place_btn.setGeometry(200, 90, 150, 40)
        self.place_btn.clicked.connect(self.place_object)
        
        # 🤖 자동 실행 버튼 추가
        self.auto_btn = QPushButton("자동 실행", self)
        self.auto_btn.setGeometry(370, 90, 150, 40)
        self.auto_btn.clicked.connect(self.auto_run)

        # ⏲️ 타이머 설정 (100ms마다 update_frame 호출 → 실시간 카메라 처리)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        # 📌 각 위치(A~D)로 이동하기 위한 조인트 각도 사전 정의
        self.move_coords_to_angles = {
            4: [-65.15, 8.17, -75.56, -8, 93.86, -10],       # D 위치 (예: 가장 왼쪽)
            3: [-26, -33.92, -30.75, 0.66, 90.08, -155],     # C 위치
            1: [54.58, -42.89, -11.16, -12.3, 90.61, -80],   # A 위치
            2: [103.18, 9.75, -75.32, -11.16, 90.76, -30],   # B 위치
        }

        # UI 창 보이기
        self.show()

    # ------------------------------------------------------------
    # 📌 1. 홈 위치 이동 버튼 동작: 그리퍼 닫고, 초기 자세로 이동
    # ------------------------------------------------------------
    def go_home_position(self):
        for _ in range(4):  # 그리퍼를 여러 번 닫아 안정적으로 잡도록 설정
            self.mycobot.set_pro_gripper_close(14)
            time.sleep(2)

        home_angles = [0.0, 45.0, -90.0, -45.0, 90.0, -90.0]  # 표준 초기 각도
        self.mycobot.send_angles(home_angles, 30)
        self.status_label.setText("✅ 홈 위치로 이동 완료")

    # ------------------------------------------------------------
    # 📌 2. 객체 위로 이동 버튼 동작
    # ------------------------------------------------------------
    def move_above_object(self):
        if self.latest_coords:
            self.locked_coords = self.latest_coords  # 현재 좌표를 고정
            self.update_enabled = False  # 실시간 업데이트 일시 중단

        if self.locked_coords and self.roi_coords:
            x, y = self.locked_coords
            x1, y1, x2, y2 = self.roi_coords
            roi_center_x = (x1 + x2) // 2
            roi_center_y = (y1 + y2) // 2
            roi_width = x2 - x1
            roi_height = y2 - y1

            try:
                # ArUco 마커 2개를 기준으로 회전 각도(theta) 계산
                pts = self.roi_marker_pts
                dx_vec = pts[1][0] - pts[0][0]
                dy_vec = pts[1][1] - pts[0][1]
                theta = math.atan2(dy_vec, dx_vec)  # 탄젠트 각도 계산
            except:
                self.status_label.setText("❌ ROI 회전각 계산 실패")
                return

            # ROI 중심 기준 객체의 상대 위치 (픽셀)
            dx_pixel = x - roi_center_x
            dy_pixel = y - roi_center_y

            # 회전 보정: 회전된 ROI 기준으로 좌표 변환
            dx_rot = dx_pixel * math.cos(theta) - dy_pixel * math.sin(theta)
            dy_rot = -dx_pixel * math.sin(theta) - dy_pixel * math.cos(theta)

            # 4분면 보정: 방향별 감도 차이 보정
            if x >= roi_center_x and y <= roi_center_y:
                dx_rot *= -0.1
                dy_rot *= 0.8
            elif x < roi_center_x and y <= roi_center_y:
                dx_rot *= 0.8
                dy_rot *= -1.5
            elif x < roi_center_x and y > roi_center_y:
                dx_rot *= -1.5
                dy_rot *= 0.5
            elif x >= roi_center_x and y > roi_center_y:
                dx_rot *= 0.8
                dy_rot *= 1.5

            # 픽셀 → mm 변환 (ROI 크기 기준으로 비례식 적용)
            scale_x = 200.0 / roi_width
            scale_y = 200.0 / roi_height
            dx_mm = dx_rot * scale_x
            dy_mm = dy_rot * scale_y

            # 로봇 이동 위치 계산
            robot_x = 250.0 + dx_mm  # 기준점 250mm 기준
            robot_y = 0.0 + dy_mm

            target_coords = [robot_x, robot_y, 280.0, 180.0, 0.0, 0.0]  # Z고정
            self.move_coords = target_coords  # 다음 이동에 사용
            self.mycobot.send_coords(target_coords, 30, 0)

            print(f"[좌표 전송] X={robot_x:.1f}, Y={robot_y:.1f}, Z=280")
            self.status_label.setText(f"🤖 ROI 회전보정 이동: X={robot_x:.1f}, Y={robot_y:.1f}")

            # 도착 대기 후 그리퍼 열기
            self.update_enabled = True
            self.wait_until_arrival(target_coords)
            for _ in range(4):
                self.mycobot.set_pro_gripper_open(14)
                time.sleep(2)
        else:
            self.status_label.setText("❌ ROI 또는 객체 중심 좌표 없음")

    # ------------------------------------------------------------
    # 📌 3. 픽업 버튼 동작: Z축 낮추고 그리퍼로 물체 잡기
    # ------------------------------------------------------------
    def pickup_object(self):
        pickup_coords = self.move_coords.copy()
        pickup_coords[2] = max(pickup_coords[2] - 110, 100)  # 너무 낮아지지 않도록 최소값 제한
        self.mycobot.send_coords(pickup_coords, 40, 0)
        self.wait_until_arrival(pickup_coords)

        for _ in range(4):  # 물체를 안정적으로 잡기 위해 4번 시도
            self.mycobot.set_pro_gripper_close(14)
            time.sleep(2)

        self.mycobot.send_coords(self.move_coords, 40, 0)  # 다시 원래 높이로 복귀

    # ------------------------------------------------------------
    # 📌 4. 플레이스 버튼 동작: 드롭다운 위치로 이동 후 놓기
    # ------------------------------------------------------------
    def place_object(self):
        target = self.place_combo.currentText()  # 선택된 위치 A/B/C/D
        index = {"D": 4, "C": 3, "A": 1, "B": 2}[target]
        target_angles = self.move_coords_to_angles[index]
        self.mycobot.send_angles(target_angles, 40)
        time.sleep(3)

        self.status_label.setText(f"📦 {target} 위치로 배치 완료")
        time.sleep(3)

        for _ in range(4):  # 그리퍼 열기
            self.mycobot.set_pro_gripper_open(14)
            time.sleep(2)

    # ------------------------------------------------------------
    # 📌 로봇 좌표 도착 여부 체크
    # ------------------------------------------------------------
    def wait_until_arrival(self, target, tol=5.0):
        for _ in range(30):
            now = self.mycobot.get_coords()
            if now and all(abs(now[i] - target[i]) < tol for i in range(3)):
                return True
            time.sleep(0.3)
        return False

    
    def update_frame(self):
        # 📸 카메라에서 프레임 한 장 캡처
        ret, frame = self.cap.read()
        if not ret:
            return  # 프레임 읽기에 실패하면 아무것도 하지 않음

        frame_h, frame_w = frame.shape[:2]  # 프레임의 높이와 너비 추출

        # 🔍 ArUco 마커 탐지를 위한 사전 정의된 딕셔너리 및 파라미터 설정
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # 6x6 마커 사용
        parameters = cv2.aruco.DetectorParameters()  # 기본 파라미터 사용

        # 🔎 프레임에서 ArUco 마커 탐지 (마커 경계점 좌표와 ID 반환)
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        # ✅ 2개 이상의 마커가 감지되었을 때만 ROI를 설정함
        if ids is not None and len(ids) >= 2:
            # corners → 마커 하나당 4개 꼭짓점 좌표가 2차원 배열로 들어있음
            corners = [c[0] for c in corners]

            # 각 마커의 중심좌표를 계산한 후, x, y순으로 정렬
            pts = sorted([c.mean(axis=0) for c in corners], key=lambda p: (p[0], p[1]))
            self.roi_marker_pts = pts  # 나중에 회전 각도 계산에 사용

            # 좌측 상단 마커와 우측 하단 마커 좌표를 추출 (ROI 범위 계산용)
            top_left = tuple(pts[0].astype(int))
            bottom_right = tuple(pts[-1].astype(int))

            x1, y1 = top_left
            x2, y2 = bottom_right
            x1, x2 = sorted([x1, x2])  # 혹시 좌우가 반대로 들어올 경우 정렬
            y1, y2 = sorted([y1, y2])  # 혹시 상하가 반대로 들어올 경우 정렬

            # 여유 여백을 조금 더 줘서 ROI 영역 확장
            margin = 30
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_w, x2 + margin)
            y2 = min(frame_h, y2 + margin)
            self.roi_coords = (x1, y1, x2, y2)  # ROI 영역 저장

            # ROI 영역만 잘라냄
            roi = frame[y1:y2, x1:x2]

            # 🔄 ROI를 HSV 색공간으로 변환 (색상 검출에 유리)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 🎨 HSV 범위별 색상 목록 정의 (각 물체 색에 따라 조절 가능)
            hsv_ranges = [
                ("red",    np.array([0, 100, 100]),    np.array([10, 255, 255])),
                ("orange", np.array([11, 100, 100]),   np.array([20, 255, 255])),
                ("yellow", np.array([15, 80, 80]),     np.array([40, 255, 255])),
                ("green",  np.array([45, 100, 100]),   np.array([75, 255, 255])),
                ("sky",    np.array([76, 100, 100]),   np.array([95, 255, 255])),
                ("blue",   np.array([100, 100, 100]),  np.array([130, 255, 255])),
                ("pupple", np.array([131, 100, 100]),  np.array([160, 255, 255])),
                ("pink",   np.array([161, 100, 100]),  np.array([170, 255, 255])),
                ("brown",  np.array([10, 150, 20]),    np.array([20, 200, 200])),
                ("black",  np.array([0, 0, 0]),        np.array([180, 255, 50]))
            ]

            found = False  # 객체를 찾았는지 여부

            # 🎯 각 색상 범위별로 마스크 → 컨투어 → 중심좌표 계산
            for color_name, lower, upper in hsv_ranges:
                mask = cv2.inRange(hsv, lower, upper)  # 해당 색상만 1, 나머지는 0

                # 외곽선(윤곽선) 추출
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    # 너무 작은 물체는 제외
                    if cv2.contourArea(cnt) < 200:
                        continue

                    # 중심좌표 계산 (무게중심 이용)
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue

                    cx = int(M["m10"] / M["m00"])  # x 중심
                    cy = int(M["m01"] / M["m00"])  # y 중심

                    # ROI 기준 → 전체 프레임 기준으로 보정
                    full_cx = cx + x1
                    full_cy = cy + y1

                    # 실시간 업데이트가 가능하면 latest_coords를 갱신
                    if self.update_enabled:
                        self.detected_color_name = color_name  # <- 객체의 색상 이름 저장
                        self.latest_coords = (full_cx, full_cy)

                    # 화면에 객체 중심 표시 (녹색 원 + 색상 이름)
                    cv2.circle(frame, (full_cx, full_cy), 6, (0, 255, 0), -1)
                    cv2.putText(frame, f"{color_name}", (full_cx + 5, full_cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # 상태 라벨에 색상과 좌표 표시
                    self.status_label.setText(f"✅ {color_name} 중심: ({full_cx}, {full_cy})")
                    found = True
                    break  # 하나만 찾고 종료
                if found:
                    break

            if not found:
                self.status_label.setText("❌ ROI 내 객체 없음")

            # ROI 표시 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "ROI", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            # 마커가 2개 미만일 경우 인식 불가
            self.status_label.setText("❌ ArUco 마커 2개 필요")

        # 📺 최종 영상 출력
        cv2.imshow("실시간 객체 중심 인식", frame)
        cv2.waitKey(1)  # OpenCV 창 유지용 (실질적 딜레이 없음)
    
     # ------------------------------------------------------------
    # 📌 시작~끝 자동화 
    # ------------------------------------------------------------
    def auto_run(self):
        # 1. 홈위치 이동
        self.go_home_position()

        self.status_label.setText("🔄 객체 인식 대기 중...")

        # 2. 객체 인식될 때까지 대기
        if not self.wait_for_object(timeout=10):
            self.status_label.setText("❌ 객체 인식 실패 (10초 내)")
            return
        self.status_label.setText("✅ 객체 인식 완료, 이동 중...")

        # 3. 객체 위로 이동
        self.move_above_object()

        # 4. 픽업
        self.pickup_object()

        color = self.detected_color_name
        if color == "yellow":
            self.place_combo.setCurrentText("A")
        elif color == "red":
            self.place_combo.setCurrentText("B")
        elif color == "green":
            self.place_combo.setCurrentText("C")
        elif color == "pupple":
            self.place_combo.setCurrentText("D")
        else:
            self.status_label.setText(f"❌ '{color}' 색상은 분류 대상 아님")
            return
        
        self.place_object()

        self.status_label.setText("🎉 자동 작업 완료")

    # 객체가 인식될 때까지 일정 시간 대기
    def wait_for_object(self, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            if self.latest_coords is not None:
                return True
            QApplication.processEvents()  # UI 이벤트 처리
            time.sleep(0.1)
        return False

    def closeEvent(self, event):
        # 📴 카메라 장치 닫기
        self.cap.release()
        # 🧹 모든 OpenCV 창 닫기 (메모리 해제)
        cv2.destroyAllWindows()
        # 부모 클래스의 closeEvent를 호출하여 종료 완료 처리
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyCobotPickupApp()
    sys.exit(app.exec_())
