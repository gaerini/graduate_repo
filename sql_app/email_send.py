import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders

def sending_email(destination, imageArray):
    # 발신자 및 수신자 이메일 설정
    sender_email = "jimmy7335@gmail.com"
    sender_password = "yxtc tdam uovz znqi"
    receiver_email = destination

    # 이메일 메시지 생성
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = '이미지 첨부된 이메일'

    # 이메일 본문 설정 (텍스트)
    body = "이메일 본문에 포함된 이미지입니다."
    message.attach(MIMEText(body, 'plain'))

    # 이미지 파일 첨부
    imageArray = imageArray[0].split(',')
    for image in imageArray:
        print(imageArray)
        image = os.path.join("/Users/ji-hokim/Documents/graduateProject/BE/sql_app/photo_stored/", image)
        with open(image, "rb") as image_file:
            image_data = MIMEImage(image_file.read(), name=image)
            message.attach(image_data)

    # SMTP 서버에 연결 및 이메일 전송
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print('이메일이 성공적으로 전송되었습니다!')
    except Exception as e:
        print(f'이메일 전송 중 오류 발생: {e}')