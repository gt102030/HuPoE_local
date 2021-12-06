from PyQt5 import QtWidgets, QtSql
from LoginUi_1 import Ui_Form
from face_r import Ui_Form as Ui_Form_face
import sys

class myApp(QtWidgets.QWidget, Ui_Form, Ui_Form_face):
    def __init__(self):
        super(myApp, self).__init__()
        self.setupUi(self)
        self.openDB()
        self.pushButton.clicked.connect(self.checkUser)
        #self.
        

    def openDB(self):
        self.db = QtSql.QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("data.sqlite")
        if not self.db.open():
            print("Error")
        self.query = QtSql.QSqlQuery()

    def checkUser(self):
        username1 = self.lineEdit.text()
        password1 = self.lineEdit_2.text()
        print(username1, password1)
        self.query.exec_("select * from userdata where username = '%s' and password = '%s';"%(username1, password1))
        self.query.first()
        if self.query.value("username") != None and self.query.value("password") != None:
            print("login successful!")
            Form.hide()
            self.secondWindow()
            
        else: 
            print("login failed!")
            win = QtWidgets.QWidget()
            QtWidgets.QMessageBox.about(win, "Error", "비밀번호 오류")
            #exit()

    def secondWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Form_face()
        self.ui.setupUi(self.window)
        self.window.show()
        print("second f_r")
        #self.Videocapture_ = 0
        #self.ui.startVideo(self.Videocapture_)
        #print("Video Played")
        

        



  
if __name__ == "__main__":
        app  = QtWidgets.QApplication(sys.argv)
        Form = myApp()
        Form.show()
        sys.exit(app.exec_())
        