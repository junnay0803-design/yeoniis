1단계: 파이썬(Python) 설치 확인
먼저 컴퓨터에 파이썬이 깔려 있어야 합니다.

2단계: 필요 외부 도구 설치 (requirements.txt)
우리가 만든 프로그램은 Streamlit이나 Plotly 같은 외부 도구를 사용합니다. 
파일 확인: 내 프로젝트 폴더 안에 아까 만든 requirements.txt를 저장해서 사용합니다.

명령어 입력: 터미널에 아래 내용을 복사해서 붙여넣고 엔터를 치세요
pip install -r requirements.txt

3단계: 프로그램 실행하기
VScode 앱에서 solar_analyzer_v2.py (메인소스코드)를 엽니다.

명령어 입력: 터미널에 아래 명령어를 입력합니다.
streamlit run solar_analyzer_v2.py