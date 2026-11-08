import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from starclass.core.observations import Source
from starclass.core.photometry import Aperture, relative_magnitude


class Window(QMainWindow):
    completed = Signal(object)
    progress = Signal(int, int)

    def __init__(self, inspect, extract, reader, export, classify, train):
        super().__init__()
        self.inspect = inspect
        self.extract = extract
        self.reader = reader
        self.export = export
        self.classify = classify
        self.train = train
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.stop = Event()
        self.path = self.meta = self.image = self.rows = self.source = self.model = None
        self.sources = []
        self.busy = False
        self.setWindowTitle("Starclass · Переменные звёзды")
        self.resize(1200, 800)
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 16)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        title = QLabel("Переменные звёзды")
        title.setObjectName("title")
        bar.addWidget(title, 1)
        self.open_button = self.button("Открыть FITS", self.choose)
        self.save_button = self.button("Экспорт CSV", self.save)
        bar.addWidget(self.open_button)
        bar.addWidget(self.save_button)
        self.info = QLabel(
            "Откройте серию TESS, чтобы найти источник и измерить его яркость."
        )
        layout.addWidget(self.info)
        split = QSplitter()
        layout.addWidget(split, 1)
        visual = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(visual)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        self.image_canvas = FigureCanvasQTAgg(Figure(layout="constrained"))
        self.image_axes = self.image_canvas.figure.subplots()
        self.image_canvas.mpl_connect("button_press_event", self.select_image)
        image_layout.addWidget(self.image_canvas)
        frame_controls = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setAccessibleName("Кадр серии")
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.change_frame)
        self.frame_label = QLabel("Опорный кадр")
        frame_controls.addWidget(self.frame_slider, 1)
        frame_controls.addWidget(self.frame_label)
        image_layout.addLayout(frame_controls)
        visual.addWidget(image_panel)
        self.curve_canvas = FigureCanvasQTAgg(Figure(layout="constrained"))
        self.curve_axes = self.curve_canvas.figure.subplots()
        visual.addWidget(self.curve_canvas)
        sidebar = QWidget()
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(320)
        panel = QVBoxLayout(sidebar)
        split.addWidget(sidebar)
        split.setStretchFactor(0, 1)
        panel.addWidget(QLabel("Источник"))
        self.targets = QComboBox()
        self.targets.setAccessibleName("Найденные источники")
        self.targets.currentIndexChanged.connect(self.select_list)
        panel.addWidget(self.targets)
        panel.addWidget(QLabel("Радиус апертуры, пиксели"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.5, 4)
        self.radius.setValue(2)
        self.radius.valueChanged.connect(self.invalidate)
        panel.addWidget(self.radius)
        self.manual = QCheckBox("Выбрать точку вручную")
        panel.addWidget(self.manual)
        self.run_button = self.button("Измерить яркость", self.analyze)
        panel.addWidget(self.run_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        panel.addWidget(self.progress_bar)
        self.cancel_button = self.button("Отменить", self.stop.set)
        panel.addWidget(self.cancel_button)
        panel.addSpacing(14)
        panel.addWidget(QLabel("Результат"))
        self.units = QComboBox()
        self.units.addItems(["Относительный поток", "Поток", "Относительная величина"])
        self.units.currentIndexChanged.connect(self.draw_curve)
        panel.addWidget(self.units)
        self.result = QLabel("После измерения здесь появится сводка.")
        self.result.setWordWrap(True)
        panel.addWidget(self.result)
        self.model_button = self.button("Открыть доверенную модель", self.choose_model)
        panel.addWidget(self.model_button)
        self.classify_button = self.button("Определить тип", self.predict)
        panel.addWidget(self.classify_button)
        panel.addStretch(1)
        file_menu = self.menuBar().addMenu("Файл")
        for label, shortcut, callback in (
            ("Открыть FITS", QKeySequence.StandardKey.Open, self.choose),
            ("Экспорт CSV", QKeySequence.StandardKey.Save, self.save),
        ):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            file_menu.addAction(action)
        self.menuBar().addMenu("Модель").addAction(
            "Обучить по манифесту…", self.choose_training
        )
        self.completed.connect(self.finish)
        self.progress.connect(self.update_progress)
        self.draw_image()
        self.draw_curve()
        self.controls()

    @staticmethod
    def button(label, callback):
        button = QPushButton(label)
        button.clicked.connect(callback)
        return button

    def controls(self):
        self.open_button.setEnabled(not self.busy)
        self.save_button.setEnabled(not self.busy and bool(self.rows))
        self.run_button.setEnabled(not self.busy and self.source is not None)
        self.cancel_button.setVisible(self.busy)
        self.progress_bar.setVisible(self.busy)
        self.targets.setEnabled(not self.busy and bool(self.sources))
        self.radius.setEnabled(not self.busy)
        self.manual.setEnabled(not self.busy)
        self.frame_slider.setEnabled(not self.busy and self.meta is not None)
        self.classify_button.setEnabled(
            not self.busy and bool(self.rows) and self.model is not None
        )
        self.model_button.setEnabled(not self.busy)

    def submit(self, work, success):
        if self.busy:
            return
        self.busy = True
        self.stop.clear()
        self.progress_bar.setRange(0, 0)
        self.controls()
        future = self.pool.submit(work)
        future.add_done_callback(lambda done: self.completed.emit((done, success)))

    def finish(self, payload):
        self.busy = False
        future, success = payload
        try:
            success(future.result())
        except InterruptedError as exc:
            self.statusBar().showMessage(str(exc), 6000)
        except Exception as exc:
            logging.getLogger(__name__).exception("Analysis worker failed")
            QMessageBox.warning(self, "Операция не завершена", str(exc))
        self.controls()

    def choose(self):
        if not self.busy:
            path, _ = QFileDialog.getOpenFileName(
                self, "Открыть серию TESS", filter="FITS (*.fits *.fit)"
            )
            if path:
                self.load(path)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if not self.busy and len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.load(event.mimeData().urls()[0].toLocalFile())

    def load(self, path):
        def loaded(result):
            self.meta, self.image, self.sources = result
            self.path = path
            self.source = None
            self.targets.blockSignals(True)
            self.targets.clear()
            self.targets.addItem("Выберите источник")
            for i, source in enumerate(self.sources, 1):
                self.targets.addItem(f"{i} · x {source.x:.1f}, y {source.y:.1f}")
            self.targets.blockSignals(False)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, len(self.meta.time))
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.frame_label.setText("Опорный кадр")
            self.info.setText(f"{self.meta.path.name}  ·  {len(self.meta.time)} кадров")
            self.invalidate()

        self.submit(lambda: self.inspect(path), loaded)

    def invalidate(self):
        self.rows = None
        self.result.setText("Выберите источник и запустите измерение.")
        self.draw_image()
        self.draw_curve()
        self.controls()

    def select_list(self, index):
        self.source = self.sources[index - 1] if index > 0 else None
        self.invalidate()

    def select_image(self, event):
        if self.busy or self.image is None or event.inaxes != self.image_axes:
            return
        x, y = event.xdata, event.ydata
        if self.manual.isChecked():
            self.targets.setCurrentIndex(0)
            self.source = Source(x, y)
            self.invalidate()
        elif self.sources:
            distances = [np.hypot(s.x - x, s.y - y) for s in self.sources]
            index = int(np.argmin(distances))
            if distances[index] <= 3:
                self.targets.setCurrentIndex(index + 1)

    def change_frame(self, index):
        if self.path is None:
            return
        if index:
            try:
                with self.reader(self.path) as (_, frame):
                    self.image = frame(index - 1)[0]
                self.frame_label.setText(
                    f"Кадр {index} · {self.meta.time[index - 1]:.5f} д"
                )
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(str(exc), 6000)
        else:
            self.image = self.reference
            self.frame_label.setText("Опорный кадр")
        self.draw_image()

    def draw_image(self):
        ax = self.image_axes
        ax.clear()
        if self.image is not None:
            if self.frame_slider.value() == 0:
                self.reference = self.image
            finite = self.image[np.isfinite(self.image)]
            if finite.size:
                low, high = np.percentile(finite, [5, 99])
                ax.imshow(
                    np.arcsinh((self.image - low) / max(high - low, 1e-12)),
                    cmap="gray",
                    origin="lower",
                    interpolation="nearest",
                )
                for index, source in enumerate(self.sources, 1):
                    ax.add_patch(
                        Circle((source.x, source.y), 0.65, fill=False, color="#a5cbff")
                    )
                    ax.annotate(str(index), (source.x + 0.7, source.y), color="#a5cbff")
                if self.source:
                    for radius, style in (
                        (self.radius.value(), "-"),
                        (3, "--"),
                        (5, "--"),
                    ):
                        ax.add_patch(
                            Circle(
                                (self.source.x, self.source.y),
                                radius,
                                fill=False,
                                color="#fbc76d",
                                linestyle=style,
                            )
                        )
                ax.set(xlabel="x, пиксели", ylabel="y, пиксели")
        else:
            ax.text(0.5, 0.5, "Снимок появится после загрузки FITS", ha="center")
            ax.set_axis_off()
        self.image_canvas.draw_idle()

    def analyze(self):
        path, source = self.path, self.source
        aperture = Aperture(
            radius=self.radius.value(),
            sky_inner=self.radius.value() + 1,
            sky_outer=self.radius.value() + 3,
        )
        self.submit(
            lambda: self.extract(
                path,
                source,
                aperture,
                neighbors=self.sources,
                progress=self.progress.emit,
                cancelled=self.stop.is_set,
            ),
            self.show_analysis,
        )

    def update_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def show_analysis(self, outcome):
        _, self.rows = outcome
        valid = sum(not row.reason for row in self.rows)
        self.result.setText(
            f"Пригодных кадров: {valid} из {len(self.rows)}."
            + (" Проверьте апертуру и качество кадров." if not valid else "")
        )
        self.draw_curve()

    def draw_curve(self):
        ax = self.curve_axes
        ax.clear()
        rows = [row for row in self.rows or [] if not row.reason]
        if rows:
            time = np.array([row.time for row in rows])
            flux = np.array([row.flux for row in rows])
            median = np.median(flux)
            mode = self.units.currentIndex()
            values = (flux / median, flux, relative_magnitude(flux, median))[mode]
            ax.scatter(time - time[0], values, s=8, color="#3157d5")
            ax.set(xlabel="Дни от первого измерения", ylabel=self.units.currentText())
            ax.grid(alpha=0.2)
            if mode == 2:
                ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "Нет измерений для отображения", ha="center")
            ax.set_axis_off()
        self.curve_canvas.draw_idle()

    def save(self):
        if self.busy or not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт кривой", "lightcurve.csv", "CSV (*.csv)"
        )
        if path:
            self.export(path, self.rows)
            self.statusBar().showMessage("Кривая сохранена", 6000)

    def choose_model(self):
        path = QFileDialog.getExistingDirectory(self, "Папка доверенной модели")
        if (
            path
            and QMessageBox.question(
                self,
                "Доверие к модели",
                "Загружайте только модель, созданную вами. Продолжить?",
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.model = path
            self.controls()

    def predict(self):
        if self.radius.value() != Aperture().radius:
            QMessageBox.information(
                self,
                "Настройки модели",
                "Для модели измерьте кривую с радиусом 2 пикселя.",
            )
            return

        def show(result):
            label = result["label"] or "Неуверенная классификация"
            scores = sorted(result["scores"].items(), key=lambda item: -item[1])
            self.result.setText(
                label + "\n\n" + "\n".join(f"{k} · {v:.0%}" for k, v in scores)
            )

        self.submit(lambda: self.classify(self.model, self.rows), show)

    def choose_training(self):
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Манифест обучения", filter="CSV (*.csv)"
        )
        if path:
            output, _ = QFileDialog.getSaveFileName(self, "Новая папка модели")
            if output:

                def trained(meta):
                    self.model = output
                    score = meta["metrics"]["rf"]["test"]["report"]["macro avg"][
                        "f1-score"
                    ]
                    QMessageBox.information(
                        self, "Модель обучена", f"Macro-F1 на тесте: {score:.3f}"
                    )

                self.submit(lambda: self.train(path, output, self.stop.is_set), trained)

    def closeEvent(self, event):
        if self.busy:
            self.stop.set()
            event.ignore()
        else:
            self.pool.shutdown(wait=False)
            event.accept()


def launch(*dependencies):
    app = QApplication([])
    app.setStyleSheet("""
        QMainWindow { background: #f6f7f9; }
        QLabel { color: #17212b; }
        QLabel#title { font-size: 24px; font-weight: 650; }
        QPushButton, QComboBox, QDoubleSpinBox { padding: 7px 10px; }
        QPushButton:enabled { color: #213c9b; }
    """)
    window = Window(*dependencies)
    window.show()
    return app.exec()
