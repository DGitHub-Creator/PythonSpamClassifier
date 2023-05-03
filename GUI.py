#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2023 DZX.
#
#  All rights reserved.
#
#  This software is protected by copyright law and international treaties. No part of this software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright owner.
#
#  For permission requests, please contact the copyright owner at the address below.
#
#  DZX
#
#  xindemicro@outlook.com
#

# 导入相关库
import base64
import ctypes
import io
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor, QPainter, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMenu, QAction, QPushButton, QRadioButton,
    QSizePolicy, QSystemTrayIcon, QTextEdit, QVBoxLayout, QWidget, QMessageBox
)
from wordcloud import WordCloud

from src import single_classification as model_run

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 将导入 nltk 和 wordcloud 相关的代码移动到对应函数中
# 避免一开始就加载这些库，从而加快启动速度
def lancaster_tokenizer(text):
    from nltk import LancasterStemmer, word_tokenize

    """
    将一个字符串分词并进行词干提取。
    :param text:文本
    :return:
    """
    # 初始化词干提取器
    lancaster_stemmer = LancasterStemmer()
    # 分词并将每个单词进行词干提取，并返回处理后的结果
    return [
        lancaster_stemmer.stem(token) for token in word_tokenize(text.lower())
    ]


def get_stop_words():
    from nltk.corpus import stopwords
    import string

    # 获取停用词
    stop_words = list(
        lancaster_tokenizer(
            " ".join(stopwords.words("english") + list(string.punctuation))))
    return stop_words


stop_words = None
tokenizer = None


class EmittingStream(io.StringIO):

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, text):
        try:
            self.callback(text)
        except Exception as e:
            logging.error(f"Error in callback function: {e}")
            traceback.print_exc()
        finally:
            super().write(text)


class WordCloudWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        icon = QIcon(r'icon\word_cloud_icon.png')
        self.setWindowIcon(icon)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # self.wordcloud_prompt = QLabel("邮件词云图：")
        # self.wordcloud_prompt.setAlignment(Qt.AlignCenter)
        # layout.addWidget(self.wordcloud_prompt)

        self.wordcloud_output = QLabel()
        layout.addWidget(self.wordcloud_output)

        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()
        self.setGeometry(geometry.width() // 4, geometry.height() // 4,
                         geometry.width() // 4, geometry.height() // 4)
        self.setWindowTitle("词云图")

    def display_wordcloud(self, img_data):
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(img_data))
        self.wordcloud_output.setPixmap(pixmap)

        self.wordcloud_output.setScaledContents(True)
        self.wordcloud_output.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # 调整窗口大小以适应词云图的大小
        width = int(pixmap.width() * 1.1)
        height = int(pixmap.height() * 1.1)
        self.resize(width, height)


class AboutBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)

    def closeEvent(self, event):
        event.accept()


class LogOutput(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def append(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # 设置主界面的显示状态变量
        self.is_visible = True
        # 创建一个空的非模态QMessageBox对象
        self.about_box = None
        self.initUI()
        # 重定向输出流到self.log_output
        sys.stdout = EmittingStream(self.write_output)

    def write_output(self, text):
        self.log_output.append(text)

    def initUI(self):
        icon = QIcon(r'icon\spam_icon.png')
        self.setWindowIcon(icon)

        self.icon = QIcon(r'icon\icon.png')
        self.systray = QSystemTrayIcon(self.icon, None)

        # 创建系统托盘图标
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon(r'icon\icon.png'))
        self.tray_icon.setVisible(True)

        # 创建系统托盘菜单
        tray_menu = QMenu(self)
        self.show_action = QAction('隐藏', self)
        self.show_action.triggered.connect(self.toggleMainWindowVisibility)
        tray_menu.addAction(self.show_action)

        self.about_action = QAction('关于', self)
        self.about_action.triggered.connect(self.showAbout)
        tray_menu.addAction(self.about_action)

        self.quit_action = QAction('退出', self)
        self.quit_action.triggered.connect(QApplication.quit)
        tray_menu.addAction(self.quit_action)

        self.tray_icon.setContextMenu(tray_menu)

        # 创建主菜单
        main_menu = self.menuBar()

        # 创建"Help"菜单
        help_menu = main_menu.addMenu('帮助')

        # 创建"About"菜单项
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.showAbout)
        help_menu.addAction(about_action)

        # 将系统托盘图标的activated信号连接到trayIconActivated方法
        # self.tray_icon.activated.connect(self.trayIconActivated)
        self.tray_icon.activated.connect(
            lambda reason: self.toggleMainWindowVisibility() if reason == QSystemTrayIcon.Trigger else None)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # 输入框和按钮
        self.file_input = QLineEdit()
        self.content_input = QTextEdit()
        self.content_input.setEnabled(False)
        self.choose_file_button = QPushButton("选择文件")
        self.choose_file_button.clicked.connect(self.choose_file)
        self.choose_file_button.setFixedSize(150, 30)

        # 选择输入框按钮
        self.file_radio = QRadioButton()
        self.file_radio.setChecked(True)
        self.content_radio = QRadioButton()
        self.file_radio.toggled.connect(self.toggle_file_input)
        self.content_radio.toggled.connect(self.toggle_content_input)

        # 下拉框
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItem("Naive Bayes")
        self.model_combo_box.addItem("LSTM")
        self.model_combo_box.addItem("Transformer")
        self.model_combo_box.setFixedSize(150, 30)

        # 输出框
        self.log_output = LogOutput(central_widget)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("border: 1px solid black;"
                                      "font-size: 20px;")  # 在这里设置字体大小
        self.label_output = QLabel()
        self.label_output.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.label_output.setStyleSheet("border: 1px solid black;")
        self.wordcloud_output = QLabel()

        # 运行按钮
        self.run_button = QPushButton("运行")
        self.run_button.clicked.connect(self.run)
        self.run_button.setFixedSize(150, 30)

        # 布局设置
        file_prompt = QLabel("文件路径：")
        content_prompt = QLabel("邮件内容：")

        file_layout = QHBoxLayout()
        file_layout.addWidget(file_prompt)
        file_layout.addWidget(self.file_radio)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.choose_file_button)
        layout.addLayout(file_layout)

        content_layout = QHBoxLayout()
        content_layout.addWidget(content_prompt)
        content_layout.addWidget(self.content_radio)
        content_layout.addWidget(self.content_input)
        layout.addLayout(content_layout)

        combo_layout = QHBoxLayout()
        combo_layout.addStretch(1)
        combo_layout.addWidget(self.model_combo_box)
        combo_layout.addWidget(self.run_button)
        layout.addLayout(combo_layout)

        log_prompt = QLabel("日志输出：")
        label_prompt = QLabel("分类结果：")

        # 设置 log_output 和 label_output 的尺寸策略
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.log_output.sizePolicy().hasHeightForWidth())
        self.log_output.setSizePolicy(size_policy)
        self.label_output.setSizePolicy(size_policy)

        # 创建垂直布局并添加 log_prompt 和 log_output
        log_layout = QVBoxLayout()
        log_layout.addWidget(log_prompt)
        log_layout.addWidget(self.log_output)

        # 创建垂直布局并添加 label_prompt 和 label_output
        label_layout = QVBoxLayout()
        label_layout.addWidget(label_prompt)
        label_layout.addWidget(self.label_output)

        # 创建一个新的水平布局并添加 label_layout 和 log_layout
        outputs_layout = QHBoxLayout()
        outputs_layout.addLayout(label_layout, 2)  # 设置水平拉伸为 2，占20%空间
        outputs_layout.addLayout(log_layout, 8)  # 设置水平拉伸为 8，占80%空间
        layout.addLayout(outputs_layout)

        # layout.addWidget(wordcloud_prompt)
        layout.addWidget(self.wordcloud_output)

        screen = QApplication.primaryScreen()
        geometry = screen.availableGeometry()
        self.setGeometry(geometry.width() // 4, geometry.height() // 4,
                         geometry.width() // 2, geometry.height() // 2)
        self.setWindowTitle("垃圾邮件检测")
        self.show()

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        self.file_input.setText(file_path)

    def toggle_file_input(self, checked):
        self.file_input.setEnabled(checked)
        self.choose_file_button.setEnabled(checked)
        self.content_input.setEnabled(not checked)

    def toggle_content_input(self, checked):
        self.content_input.setEnabled(checked)
        self.file_input.setEnabled(not checked)
        self.choose_file_button.setEnabled(not checked)

    def generate_wordcloud(self, email_content):
        wc = WordCloud(max_words=80, min_font_size=10, font_step=2, background_color='white',
                       width=800, height=600)
        wc.generate(email_content)

        img_data = io.BytesIO()
        wc.to_image().save(img_data, format="PNG")
        img_data.seek(0)

        if not os.path.exists("image"):
            os.makedirs("image")
        current_time = time.strftime("%Y%m%d-%H%M%S")
        filename = "wordcloud-" + current_time + ".png"

        with open("image/" + filename, "wb") as f:
            f.write(img_data.read())

        return base64.b64encode(img_data.getvalue())

    def create_image_with_text(self, text, font_color, width, height):

        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setPen(QColor(font_color))
        font_size = (width + height) / 10
        painter.setFont(QFont("Arial", font_size))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
        painter.end()

        return pixmap

    def run(self):
        @contextmanager
        def capture_output():
            old_stdout = sys.stdout
            sys.stdout = EmittingStream(self.write_output)
            try:
                yield
            finally:
                sys.stdout = old_stdout

        with capture_output():
            model_name = self.model_combo_box.currentText()
            if self.file_radio.isChecked():
                file_path = self.file_input.text()
                email_content, email_label = model_run.run_file(file_path, model_name)
            elif self.content_radio.isChecked():
                email_content = self.content_input.toPlainText()
                email_content, email_label = model_run.run_content(email_content, model_name)
            else:
                return

        if email_label is not None:
            if email_label:
                result = "垃圾" + '\n' + "邮件"
                color = "red"
            else:
                result = "正常" + '\n' + "邮件"
                color = "green"
        else:
            result = "抱歉" + '\n' + "不能识别！"
            color = "gray"

        size = self.label_output.size()
        # print("Label size:", size.width(), "x", size.height())
        width = size.width()
        height = size.height()
        # width, height = self.label_output.width(), self.label_output.height()
        pixmap = self.create_image_with_text(result, color, width, height)
        try:
            # 清除 QLabel 中的内容
            self.label_output.clear()
            # 将图片设置为 QLabel 的背景，并设置缩放模式为 Qt.KeepAspectRatioByExpanding
            self.label_output.setPixmap(pixmap)
            self.label_output.setScaledContents(True)
            self.label_output.setStyleSheet("background-color: white;")
            self.label_output.setMinimumSize(0, 0)
        except Exception as e:
            print(e)

        if email_label is not None:
            self.label_output.setAlignment(Qt.AlignCenter)

        try:
            img_data = self.generate_wordcloud(email_content)
            self.wordcloud_window = WordCloudWindow(self)
            self.wordcloud_window.display_wordcloud(img_data)
            self.wordcloud_window.move(self.geometry().x() + self.geometry().width() * 3 / 5, self.geometry().y() + 60)
            self.wordcloud_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成词云图时出现错误：{e}")

    def resizeEvent(self, event):
        width = event.size().width()
        height = int(width / 1.618)  # 黄金比例
        self.resize(width, height)
        super().resizeEvent(event)

    def toggleMainWindowVisibility(self):
        if self.is_visible:
            self.hide()
            self.show_action.setText('显示')
        else:
            self.show()
            self.show_action.setText('隐藏')

        self.is_visible = not self.is_visible

    def showAbout(self):
        if self.about_box and self.about_box.isVisible():
            return
        self.about_box = AboutBox(self)
        message = 'Spam Classifier App\n\nVersion 1.0\n\n'
        message += 'Copyright (c) 2023 DZX. All rights reserved.\n\n'
        message += 'This is a simple spam classifier application that can classify emails as spam or non-spam based ' \
                   'on the TREC 2006 Spam Track Public Corpora dataset. '
        self.about_box.about(self, '关于', message)
        self.about_box.finished.connect(self.about_box.deleteLater)

    def closeEvent(self, event):
        # print("closeEvent", self.is_visible)
        if self.is_visible is False:
            # print("i am in closeEvent")
            event.ignore()
            return
        # 如果已经有一个非模态QMessageBox对象存在，则忽略关闭事件
        if self.about_box and self.about_box.isVisible():
            event.ignore()
            return

        # 创建一个 QMessageBox 对象
        msg_box = QMessageBox()

        # 添加退出、最小化和取消按钮
        exit_button = QPushButton('退出')
        minimize_button = QPushButton('最小化')
        cancel_button = QPushButton('取消')
        msg_box.addButton(exit_button, QMessageBox.YesRole)
        msg_box.addButton(minimize_button, QMessageBox.NoRole)
        msg_box.addButton(cancel_button, QMessageBox.RejectRole)

        # 更改退出、最小化和取消按钮的文本
        exit_button.setText('退出')
        minimize_button.setText('最小化')
        cancel_button.setText('取消')

        # 设置消息框的标题和文本
        msg_box.setWindowTitle('确认退出')
        msg_box.setText('你确定要退出吗?')

        # 显示消息框，并获取用户的响应
        reply = msg_box.exec_()
        print(reply)

        # 根据用户的响应执行相应操作
        if reply == 0:
            event.accept()
        elif reply == 1:
            self.toggleMainWindowVisibility()
            self.hide()
            self.tray_icon.showMessage('垃圾邮件检测', '该程序已最小化到系统托盘。')
            event.ignore()
        else:
            event.ignore()

    def trayIconActivated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.showNormal()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 加载停用词和分词器，以便在程序运行时再加载，而非一开始就加载
    stop_words = get_stop_words()
    tokenizer = lancaster_tokenizer
    window = MainWindow()
    sys.exit(app.exec_())
