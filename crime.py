import streamlit as st

# 앱 제목
st.title("범죄 예방 팁 제공")

# 범죄 발생 위험 시간대 버튼
if st.button("범죄 발생 위험 시간대"):
    time_input = st.text_input("이동하는 시간대가 어떻게 되나요? (예: 21:30)")
    
    if time_input:
        try:
            # 입력된 시간을 시각으로 변환
            hour, minute = map(int, time_input.split(':'))
            time_in_minutes = hour * 60 + minute
            
            # 시간대에 따른 메시지 제공
            if 21 * 60 <= time_in_minutes <= 23 * 60 + 59:
                st.write("가장 위험한 시간대이니 조심하세요!")
            elif 18 * 60 <= time_in_minutes <= 20 * 60 + 59 or 0 <= time_in_minutes <= 2 * 60 + 59:
                st.write("위험한 시간대이니 조심하세요!")
            else:
                st.write("비교적으로 마음을 놓으셔도 됩니다! 하지만 방심은 금물이에요!!")
        except ValueError:
            st.write("올바른 시간 형식으로 입력해주세요 (예: HH:MM)")

# 범죄 발생 위험 날씨 버튼
if st.button("범죄 발생 위험 날씨"):
    weather_input = st.text_input("이동할 날의 일기는 어떤가요? (어두움/흐림/맑음/비/눈)")
    
    if weather_input:
        weather_input = weather_input.strip()  # 공백 제거
        # 날씨에 따른 메시지 제공
        if weather_input == "어두움":
            st.write("조심하세요! 가장 위험한 일기입니다!")
        elif weather_input == "흐림":
            st.write("조심하세요! 위험한 일기입니다!")
        elif weather_input == "맑음":
            st.write("일상적으로 범죄가 자주 일어나는 일기입니다.")
        elif weather_input in ["비", "눈"]:
            st.write("상대적으로 범죄가 적게 일어난다고 마음을 놓아서는 안됩니다!")
        else:
            st.write("올바른 날씨 정보를 입력해주세요 (어두움/흐림/맑음/비/눈)")
