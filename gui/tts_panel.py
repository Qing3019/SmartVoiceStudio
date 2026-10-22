"""CosyVoice3 多语言 TTS 面板：紧凑布局、示例文本与语言模式联动。"""
from pathlib import Path
from tts.client import BASE_URL
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel,
    QTextEdit, QLineEdit, QComboBox, QPushButton, QFileDialog, QFrame
)


EXAMPLES = {
    "中文": "你好，欢迎使用智能人声音效台。",
    "英语": "Hello, welcome to Smart Voice Studio!",
    "日语": "コンニチハ。キョウモ ヨロシク オネガイシマス。",
    "韩语": "안녕하세요. 스마트 보이스 스튜디오에 오신 것을 환영합니다.",
    "德语": "Hallo! Willkommen bei Smart Voice Studio.",
    "西班牙语": "¡Hola! Bienvenido a Smart Voice Studio.",
    "法语": "Bonjour ! Bienvenue dans Smart Voice Studio.",
    "意大利语": "Ciao! Benvenuto in Smart Voice Studio.",
    "俄语": "Здравствуйте! Добро пожаловать в Smart Voice Studio."
}

LANGUAGE_HINTS = {
    "中文": "普通话自动使用零样本模式；选择方言后使用指令模式。",
    "英语": "请填写英文文本。默认跨语言克隆；如果发音错误，可尝试「指令模式」。",
    "日语": "CosyVoice3 官方示例建议将日文转为片假名发音；这里提供了片假名测试文本。",
    "韩语": "请使用韩文文本测试。模型可能受参考音色影响。",
    "德语": "请使用德文文本测试；如果输出中文，可尝试「指令模式」。",
    "西班牙语": "请使用西班牙文文本测试；语言选择不会自动翻译文本。",
    "法语": "请使用法文文本测试；模型输出语言仍需试听验证。",
    "意大利语": "请使用意大利文文本测试；语言选择不会自动翻译文本。",
    "俄语": "俄语模型实际表现存在差异；若仍输出中文，不代表 GUI 接口错误。"
}


class TTSPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        heading.addWidget(QLabel("输入合成文本"))
        heading.addStretch()
        self.example_button = QPushButton("填入示例")
        self.example_button.setProperty("role", "text")
        heading.addWidget(self.example_button)
        layout.addLayout(heading)

        self.text_edit = QTextEdit()
        self.text_edit.setMinimumHeight(77)
        self.text_edit.setMaximumHeight(105)
        self.text_edit.setPlaceholderText("输入需要合成的文字，建议先用 1～2 句进行测试")
        self.text_edit.setPlainText(EXAMPLES["中文"])
        layout.addWidget(self.text_edit)

        selection_grid = QGridLayout()
        selection_grid.setHorizontalSpacing(8)
        selection_grid.setVerticalSpacing(5)
        selection_grid.addWidget(QLabel("目标语言"), 0, 0)
        selection_grid.addWidget(QLabel("中文方言"), 0, 1)
        self.language_combo = QComboBox()
        self.language_combo.addItems(list(EXAMPLES))
        self.dialect_combo = QComboBox()
        self.dialect_combo.addItems(["普通话", "四川话", "广东话", "东北话", "上海话", "闽南话", "山东话", "天津话", "自定义"])
        selection_grid.addWidget(self.language_combo, 1, 0)
        selection_grid.addWidget(self.dialect_combo, 1, 1)
        layout.addLayout(selection_grid)

        self.custom_dialect = QLineEdit()
        self.custom_dialect.setPlaceholderText("自定义提示，例如：请用自然的四川话表达。")
        self.custom_dialect.hide()
        layout.addWidget(self.custom_dialect)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("推理策略"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("智能选择（推荐）", "auto")
        self.mode_combo.addItem("跨语言模式", "cross_lingual")
        self.mode_combo.addItem("语言 / 方言指令模式", "instruct")
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        self.language_hint = QLabel()
        self.language_hint.setObjectName("helperText")
        self.language_hint.setWordWrap(True)
        layout.addWidget(self.language_hint)

        reference_card = QFrame()
        reference_card.setObjectName("referenceCard")
        reference_layout = QVBoxLayout(reference_card)
        reference_layout.setContentsMargins(10, 10, 10, 10)
        reference_layout.setSpacing(6)
        reference_layout.addWidget(QLabel("参考音色 · 使用本人或已授权的 WAV"))
        reference_row = QHBoxLayout()
        self.reference_path = QLineEdit()
        self.reference_path.setPlaceholderText("参考录音 WAV 路径（推荐 5～10 秒清晰语音）")
        self.browse_button = QPushButton("浏览")
        self.browse_button.setProperty("role", "subtle")
        reference_row.addWidget(self.reference_path, 1)
        reference_row.addWidget(self.browse_button)
        reference_layout.addLayout(reference_row)
        self.reference_text = QLineEdit()
        self.reference_text.setPlaceholderText("参考录音逐字文本（普通话零样本模式必填）")
        reference_layout.addWidget(self.reference_text)
        layout.addWidget(reference_card)

        self.synthesize_button = QPushButton("生成语音并加载工程")
        self.synthesize_button.setProperty("role", "primary")
        self.synthesize_button.setMinimumHeight(40)
        layout.addWidget(self.synthesize_button)
        self.status_label = QLabel(f"本机服务 {BASE_URL} · 等待合成")
        self.status_label.setObjectName("helperText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.browse_button.clicked.connect(self.browse_file)
        self.example_button.clicked.connect(self.insert_example)
        self.language_combo.currentTextChanged.connect(self.update_language)
        self.dialect_combo.currentTextChanged.connect(self.update_dialect)
        self.mode_combo.currentIndexChanged.connect(self.update_mode_hint)
        self.update_language("中文")

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考音频", str(Path.home()), "WAV 音频 (*.wav)")
        if path:
            self.reference_path.setText(path)

    def insert_example(self):
        self.text_edit.setPlainText(EXAMPLES[self.language_combo.currentText()])

    def update_language(self, language):
        # 切换语言时自动清除上一种中文方言，避免服务端拒绝非中文 + 方言组合
        if self.text_edit.toPlainText().strip() in EXAMPLES.values() or not self.text_edit.toPlainText().strip():
            self.text_edit.setPlainText(EXAMPLES[language])
        is_chinese = language == "中文"
        self.dialect_combo.setEnabled(is_chinese)
        if not is_chinese:
            self.dialect_combo.setCurrentIndex(0)
            self.custom_dialect.hide()
        self.update_mode_hint()

    def update_dialect(self, dialect):
        self.custom_dialect.setVisible(self.language_combo.currentText() == "中文" and dialect == "自定义")
        self.update_mode_hint()

    def update_mode_hint(self, *args):
        language = self.language_combo.currentText()
        hint = LANGUAGE_HINTS.get(language, "")
        if self.mode_combo.currentData() == "instruct":
            hint += " 当前使用指令模式，结果仍取决于模型。"
        elif self.mode_combo.currentData() == "cross_lingual":
            hint += " 当前使用跨语言模式，适合参考音色克隆。"
        self.language_hint.setText(hint)

    def selected_mode(self):
        return self.mode_combo.currentData()

    def parameters(self):
        text = self.text_edit.toPlainText().strip()
        reference_wav = self.reference_path.text().strip().strip('"')
        reference_text = self.reference_text.text().strip()
        language = self.language_combo.currentText()
        dialect = self.dialect_combo.currentText() if language == "中文" else "普通话"
        if dialect == "自定义":
            dialect = self.custom_dialect.text().strip()
            if not dialect:
                raise ValueError("请先填写自定义方言提示")
        if not text:
            raise ValueError("请先输入待合成文本")
        if not reference_wav or not Path(reference_wav).is_file():
            raise ValueError("请选择存在的参考 WAV 文件")
        if Path(reference_wav).suffix.lower() != ".wav":
            raise ValueError("参考文件必须是 WAV 格式")
        mode = self.selected_mode()
        requires_reference_text = mode == "auto" and language == "中文" and dialect == "普通话"
        if requires_reference_text and not reference_text:
            raise ValueError("普通话零样本模式必须填写参考音频逐字文本")
        if language == "英语" and re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
            raise ValueError("你选择了英语，但文本含有中日韩文字。请选择英语示例或改用英文句子")
        return text, reference_wav, reference_text, language, dialect
