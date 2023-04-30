from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtCore import Qt

app = QApplication([])


def create_image_with_text(text, font_color, file_name):
    pixmap = QPixmap(200, 50)
    pixmap.fill(Qt.white)

    painter = QPainter(pixmap)
    painter.setPen(QColor(font_color))
    painter.setFont(QFont("Arial", 20))
    painter.drawText(20, 30, text)
    painter.end()

    pixmap.save(file_name)


create_image_with_text("垃圾邮件", "red", "junk_email.png")
create_image_with_text("正常邮件", "green", "normal_email.png")
create_image_with_text("抱歉，不能识别！", "gray", "unknown_email.png")

app.quit()
text1 = "垃圾邮件"
text2 = "正常邮件"
text3 = "抱歉，不能识别！"