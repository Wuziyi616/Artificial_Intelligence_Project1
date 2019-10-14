import os
import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from ui import *
from normal_tangram import image_utils as normal_image_utils
from normal_tangram import search_algorithm as normal_search_algorithm
from normal_tangram import config as normal_config

from thirteen_tangram import image_utils as thirteen_image_utils
from thirteen_tangram import search_algorithm as thirteen_search_algorithm
from thirteen_tangram import config as thirteen_config

from any_tangram import image_utils as any_image_utils
from any_tangram import search_algorithm as any_search_algorithm
from any_tangram import config as any_config

IMAGE_PATH = 'raw_img/'  # default path for loading images to solve
SAVE_PATH = 'result/final.png'  # default path to save the solutions


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.select_imageLE.setText('')

        # game mode, search method
        # 0 means standard Tangram game, 1 means 13 elements, 2 means any input
        self.game_mode = 0
        self.tangram_area = 8.
        # currently only support DFS
        self.search_method = 'DFS'

        # image and result
        self.raw_mask = None
        self.final_result = None
        self.select_image_flag = False

        # all images from database
        all_path = os.listdir(os.path.join(IMAGE_PATH, 'normal'))
        self.normal_path = [os.path.join(IMAGE_PATH, 'normal', path) for path in all_path]
        self.normal_num = len(self.normal_path)
        self.select_imageSB.setMinimum(0)
        self.select_imageSB.setMaximum(self.normal_num - 1)
        self.select_imageSB.setValue(0)
        all_path = os.listdir(os.path.join(IMAGE_PATH, 'thirteen'))
        self.thirteen_path = [os.path.join(IMAGE_PATH, 'thirteen', path) for path in all_path]
        self.thirteen_num = len(self.thirteen_path)
        all_path = os.listdir(os.path.join(IMAGE_PATH, 'any'))
        self.any_path = [os.path.join(IMAGE_PATH, 'any', path) for path in all_path]
        self.any_num = len(self.any_path)

        # connect actions and functions
        self.load_imageBtn.clicked.connect(self.load_image)
        self.game_modeCB.currentIndexChanged.connect(self.change_game_mode)
        self.search_methodCB.currentIndexChanged.connect(self.change_search_method)
        self.solveBtn.clicked.connect(self.solve_puzzle)

        # set init image
        img = QtGui.QPixmap("raw_img/Tangram.png")
        img = img.scaled(self.imageLb.size(), QtCore.Qt.KeepAspectRatio)
        self.imageLb.setPixmap(img)

    def load_image(self):
        """Load image to imageLb.
        If select_imageLE is empty then load from database.
        Else load the image that is assigned by user.
        """
        if self.select_imageLE.text() == '':
            if self.game_mode == 0:
                img_path = self.normal_path[self.select_imageSB.value()]
            elif self.game_mode == 1:
                img_path = self.thirteen_path[self.select_imageSB.value()]
            elif self.game_mode == 2:
                img_path = self.any_path[self.select_imageSB.value()]
            else:
                QMessageBox.information(self, 'info', 'Invalid args!', QMessageBox.Ok)
        else:
            img_path = self.select_imageLE.text()
            self.select_imageLE.setText('')
        img = QtGui.QPixmap(img_path)
        try:
            img = img.scaled(self.imageLb.size(), QtCore.Qt.KeepAspectRatio)
            self.imageLb.setPixmap(img)
            self.raw_mask = normal_image_utils.read_gray_image(img_path)
            self.select_image_flag = True
        except AttributeError:
            self.select_image_flag = False
            QMessageBox.information(self, 'info', 'Sorry, image not exist.', QMessageBox.Ok)

    def change_game_mode(self):
        """Change self.game_mode when self.game_modeCB is changed index."""
        self.game_mode = self.game_modeCB.currentIndex()
        if self.game_mode == 0:
            self.select_imageSB.setMinimum(0)
            self.select_imageSB.setMaximum(self.normal_num - 1)
            self.select_imageSB.setValue(0)
            self.areaSB.setValue(8)
        elif self.game_mode == 1:
            self.select_imageSB.setMinimum(0)
            self.select_imageSB.setMaximum(self.thirteen_num - 1)
            self.select_imageSB.setValue(0)
            self.areaSB.setValue(45)
        elif self.game_mode == 2:
            self.select_imageSB.setMinimum(0)
            self.select_imageSB.setMaximum(self.any_num - 1)
            self.select_imageSB.setValue(0)
            self.areaSB.setValue(8)
        else:
            QMessageBox.information(self, 'info', 'Invalid args!', QMessageBox.Ok)

    def change_search_method(self):
        """Change self.search_method when self.search_methodCB is changed index."""
        self.search_method = self.search_methodCB.currentText()

    def solve_puzzle(self):
        """Solve the Tangram puzzle when solveBtn is pushed."""
        if not self.select_image_flag:
            self.raw_mask = None
            self.final_result = None
            self.select_image_flag = False
            QMessageBox.information(self, 'info', 'Please select a problem first!', QMessageBox.Ok)
            return
        self.tangram_area = float(self.areaSB.value())
        if self.game_mode == 0:
            solver = normal_search_algorithm.TangramSolver(raw_mask=self.raw_mask,
                                                           tangram_s=self.tangram_area)
            colors = normal_config.COLORS
            image_utils = normal_image_utils
        elif self.game_mode == 1:
            solver = thirteen_search_algorithm.TangramSolver(raw_mask=self.raw_mask,
                                                             tangram_s=self.tangram_area)
            colors = thirteen_config.COLORS
            image_utils = thirteen_image_utils
        elif self.game_mode == 2:
            solver = any_search_algorithm.TangramSolver(raw_mask=self.raw_mask,
                                                        tangram_s=self.tangram_area)
            colors = any_config.COLORS
            image_utils = any_image_utils
        else:
            QMessageBox.information(self, 'info', 'Invalid args!', QMessageBox.Ok)

        if self.search_method.lower() == 'DFS'.lower():
            try:
                result = solver.DFS()
            except AssertionError:
                self.raw_mask = None
                self.final_result = None
                self.select_image_flag = False
                QMessageBox.information(self, 'info', 'Sorry, fail to solve the puzzle.', QMessageBox.Ok)
                return
        else:
            QMessageBox.information(self, 'info', 'Invalid args!', QMessageBox.Ok)
        if result is None:  # can't solve the puzzle
            QMessageBox.information(self, 'info', 'Sorry, fail to solve the puzzle.', QMessageBox.Ok)
        else:
            elements = [element['element'] for element in result.used_elements]
            if self.game_mode == 0:
                draw_place = self.raw_mask
            elif self.game_mode == 1:
                draw_place = result.grid
            elif self.game_mode == 2:
                draw_place = solver.mask.raw_mask
            else:
                QMessageBox.information(self, 'info', 'Invalid args!', QMessageBox.Ok)
            self.final_result = image_utils.get_final_result(draw_place, elements, colors)
            image_utils.save_image(self.final_result, SAVE_PATH)

            QMessageBox.information(self, 'info', 'Successfully solve the puzzle!', QMessageBox.Ok)
            img = QtGui.QPixmap(SAVE_PATH)
            img = img.scaled(self.imageLb.size(), QtCore.Qt.KeepAspectRatio)
            self.imageLb.setPixmap(img)
        self.raw_mask = None
        self.final_result = None
        self.select_image_flag = False


if __name__ == '__main__':
    if not os.path.exists('result/'):
        os.makedirs('result/')
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
