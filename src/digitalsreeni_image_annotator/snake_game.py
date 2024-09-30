import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QMessageBox
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QTimer

class SnakeGame(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Secret Snake Game')
        self.setFixedSize(600, 400)  # Increased size
        self.center()
        
        self.snake = [(300, 200), (290, 200), (280, 200)]
        self.direction = 'RIGHT'
        self.food = self.place_food()
        self.score = 0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)
        self.timer.start(100)
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.show()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw snake
        painter.setBrush(QColor(0, 255, 0))
        for segment in self.snake:
            painter.drawRect(segment[0], segment[1], 10, 10)
        
        # Draw food
        painter.setBrush(QColor(255, 0, 0))
        painter.drawRect(self.food[0], self.food[1], 10, 10)
        
        # Draw score
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(10, 20, f"Score: {self.score}")
        
    def keyPressEvent(self, event):
        key = event.key()
        
        if key == Qt.Key_Left and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif key == Qt.Key_Right and self.direction != 'LEFT':
            self.direction = 'RIGHT'
        elif key == Qt.Key_Up and self.direction != 'DOWN':
            self.direction = 'UP'
        elif key == Qt.Key_Down and self.direction != 'UP':
            self.direction = 'DOWN'
        elif key == Qt.Key_Escape:
            self.close()
        
    def update_game(self):
        head = self.snake[0]
        
        if self.direction == 'LEFT':
            new_head = (head[0] - 10, head[1])
        elif self.direction == 'RIGHT':
            new_head = (head[0] + 10, head[1])
        elif self.direction == 'UP':
            new_head = (head[0], head[1] - 10)
        else:  # DOWN
            new_head = (head[0], head[1] + 10)
        
        # Check if snake hit the edge
        if (new_head[0] < 0 or new_head[0] >= 600 or
            new_head[1] < 0 or new_head[1] >= 400):
            self.game_over()
            return
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 1
            self.food = self.place_food()
        else:
            self.snake.pop()
        
        if new_head in self.snake[1:]:
            self.game_over()
            return
        
        self.update()
        
    def place_food(self):
        while True:
            x = random.randint(0, 59) * 10
            y = random.randint(0, 39) * 10
            if (x, y) not in self.snake:
                return (x, y)
    
    def game_over(self):
        self.timer.stop()
        QMessageBox.information(self, "Game Over", f"Your score: {self.score}")
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SnakeGame()
    sys.exit(app.exec_())