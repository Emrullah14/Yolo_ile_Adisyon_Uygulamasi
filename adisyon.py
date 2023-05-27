from PySide2 import QtWidgets, QtGui, QtCore
import ui
import sys
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt') 
model.to(device)
class_names = model.model.names
prices = {class_names[0]:15,class_names[1]:120, class_names[2]:50, class_names[3]:70,class_names[4]:80, class_names[5]:70, class_names[6]:20, class_names[7]:60, class_names[8]:50, class_names[9]:40, class_names[10]:90, class_names[11]:60 }
cap_id=0
class Thread(QtCore.QThread):
    frame_update_signal = QtCore.Signal(np.ndarray)
    add_list_signal=QtCore.Signal(str)
    @QtCore.Slot(int)
    def __init__(self,parent) -> None:
        super(Thread,self).__init__(parent)
        self.cap = cv2.VideoCapture(cap_id)

    """
    Bu bölümde kameradan görüntü okuma, nesne tanıma ve sinyal üzerinden diğer sınıftaki fonksiyona işlenen verinin aktarımı yapılmaktadır.
    Bu fonksiyon sonsuz bir döngü ile çalışmaktadır.
    Bunun sebebi ekrandan okunan görüntünün sürekli işlenmesi ve ekranda gösterilebilmesidir.
    """
    def run(self):
        
        while True:
            # Kameradan alınan görüntüleri okuyoruz.
            ret, frame = self.cap.read()
            
            # OpenCV görüntüleri BGR (Blue, Green, Red) formatında okur. Bunu RGB (Red, Green, Blue) formatına dönüştürüyoruz.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Görüntüyü 640 x 480 çözünürlüğüne düşürüyoruz.
            # rs_frame=cv2.resize(cv2image,(640,480))
            # Ekrandaki görüntü için YOLO modeli ile tahminleme işlemi yapıyoruz. Verbose değeri konsolda belli başlı bilgileri vermemesi için kullanılmıştır.
            # Bu bilgiler önemsiz olduğu için kapatılmıştır. Verbose=True yapıldığında konsolda birçok bilgi akacaktır.
            results=model.predict(frame,verbose=False)
            for result in results:
                nd_result=result.boxes.xyxy.cpu().numpy() # Algılanan nesnenin sınır noktalarını alıyoruz.
                classes=result.boxes.cls # Algılanan nesnenin sınıf etiketlerini alıyoruz. Bu etiketler= [0, 1, 2, 3...] şeklindedir
                conf=result.boxes.conf # Algılanan nesnenin güven skorunu alıyoruz.
                for bbox in zip(nd_result,classes, conf): # Algılanan nesnenin sınır noktaları, sınıf etiketleri ve güven skoru (confusion score) içerisinde dönüyoruz.
                    # Algılanan nesnenin sınır noktalarının değerlerinin alındığı alan
                    x1,y1,x2,y2=int(bbox[0][0]),int(bbox[0][1]),int(bbox[0][2]),int(bbox[0][3])
                    # Algılanan nesnenin koordinatlarına dikdörtgen çizim işlemi yapılmaktadır.
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    #Algılanan nesnenin ismi ve modelden elde edilen güven skoru (confusion score) yazılmaktadır.
                    cv2.putText(frame,f'{class_names[int(bbox[1])]} {bbox[2]:.2f}',(x1,y1-30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1,cv2.LINE_AA)
                    # Sinyal ile algılanan nesnenin adı diğer fonksiyona gönderilmektedir.
                    self.add_list_signal.emit(f'{class_names[int(bbox[1])]} {prices[class_names[int(bbox[1])]]}₺')
            # Görüntünün X eksenine göre simetriği alınmaktadır.
            frame_t = frame.transpose((1, 0, 2)) 
            # # Görüntünün Y eksenine göre simetriği alınmaktadır.
            # frame_f = cv2.flip(frame_t, 0)
            # Görüntü sinyal ile diğer fonksiyona gönderilmektedir. Bu işlem her iterasyonda bir gerçekleşmektedir.
            self.frame_update_signal.emit(frame_t)


class RestorantApp(ui.Ui_MainWindow,QtWidgets.QMainWindow):
    def __init__(self,parent=None) -> None:
        """
        Program ilk çalıştıığında yapılması gereken işlemlerin yapıldığı kısım
        """
        super(RestorantApp,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Adisyon Uygulaması')# Projenin başlığı belirleniyor
        self.resize(1450,880)# Ekran boyutu belirleniyor
        self.ImageView.ui.histogram.hide() # Ekranda görüntünün gösterildiği nesnenin bazı gereksiz özellikleri gizleniyor.
        self.ImageView.ui.roiBtn.hide()
        self.ImageView.ui.menuBtn.hide()
        self.ImageView.view.setMouseEnabled(False,False)
        # Hesaplama sonuçlarının yazdırıldığı labellar
        self.result_labels=[self.result,self.result_1,self.result_2,self.result_3,self.result_4,self.result_5,self.result_6,self.result_7,self.result_8,self.result_9,self.result_10,self.result_11,]
        # Yemek adlarının yer aldığı dictionary
        self.cook_names={0:'yayla_corbasi',1:'mercimek_corbasi',2:'kelle_paca_corbasi',3:'ezogelin_corbasi',4:'kuru_fasulye',5:'karniyarik',6:'izmir_kofte',
                         7:'tas_kebap',8:'baklava',9:'sutlac',10:'kola',11:'ayran'}
        # Yemek sayılarının girildiği metin kutuları
        self.linedits=[self.lineEdit, self.lineEdit_1, self.lineEdit_2, self.lineEdit_3, self.lineEdit_4, self.lineEdit_5, self.lineEdit_6,
                        self.lineEdit_7, self.lineEdit_8, self.lineEdit_9, self.lineEdit_10, self.lineEdit_11]
        # Masa isimlerinin yazdığı butonlar
        self.table_buttons=[self.masa1,self.masa2,self.masa3,self.masa4,self.masa5,self.masa6,self.masa7,
                            self.masa8,self.masa9,self.masa10,self.masa11,self.masa12,self.masa13,self.masa14,self.masa15,self.masa16,self.masa17,self.masa18,self.masa19,self.masa20]
        # Yemek fiyatları
        self.prices={self.cook_names[0]:60,self.cook_names[1]:50,self.cook_names[2]:70,self.cook_names[3]:50,self.cook_names[4]:60,self.cook_names[5]:80,self.cook_names[6]:70,
                     self.cook_names[7]:90,self.cook_names[8]:120,self.cook_names[9]:40,self.cook_names[10]:20,self.cook_names[11]:15}
        # Program çalıştığında pasif olması gereken nesnelerin belirlenmesini sağlayan fonksiyon çalıştırılıyor.
        self.set_enabled(False)
        # Yukarıda oluşturulan Thread sınıfı çağırılıyor. Ekrandaki görüntü Thread sayesinde gösterilmektedir.
        self.th = Thread(self)
        # Görüntünün verildiği sinyal
        self.th.frame_update_signal.connect(lambda p: self.process_frame(p))
        # Algılanan nesnelerin alındığı sinyal
        self.th.add_list_signal.connect(lambda t: self.add_list(t))
        # Ekrandaki butonların işlemlerinin çalışması için gerekli fonksiyonun çalıştırılması.
        self.button_actions()
        
    """ Nesnelerin ekranda pasif veya aktif olmalarını sağlamak için gerekli fonksiyon"""    
    def set_enabled(self,status):
        self.listWidget.setEnabled(status)
        self.pushButton.setEnabled(status)
        for i in self.linedits:
            i.setEnabled(status)
        self.pushButton_2.setEnabled(status)
        self.pushButton_3.setEnabled(status)

    """Butonların üzerine tıklandığında yapması gereken işlemlerin tanımlandığı fonksiyon"""
    def button_actions(self):
        # Temizle butonu
        self.pushButton_3.clicked.connect(self.clear_list)
        # Gönder butonu
        self.pushButton_2.clicked.connect(self.update_entry)
        # Hesapla butonu
        self.pushButton.clicked.connect(self.calculate)
        # Masa butonları
        self.masa1.clicked.connect(lambda: self.select_table(0))
        self.masa2.clicked.connect(lambda: self.select_table(1))
        self.masa3.clicked.connect(lambda: self.select_table(2))
        self.masa4.clicked.connect(lambda: self.select_table(3))
        self.masa5.clicked.connect(lambda: self.select_table(4))
        self.masa6.clicked.connect(lambda: self.select_table(5))
        self.masa7.clicked.connect(lambda: self.select_table(6))
        self.masa8.clicked.connect(lambda: self.select_table(7))
        self.masa9.clicked.connect(lambda: self.select_table(8))
        self.masa10.clicked.connect(lambda: self.select_table(9))
        self.masa11.clicked.connect(lambda: self.select_table(10))
        self.masa12.clicked.connect(lambda: self.select_table(11))
        self.masa13.clicked.connect(lambda: self.select_table(12))
        self.masa14.clicked.connect(lambda: self.select_table(13))
        self.masa15.clicked.connect(lambda: self.select_table(14))
        self.masa16.clicked.connect(lambda: self.select_table(15))
        self.masa17.clicked.connect(lambda: self.select_table(16))
        self.masa18.clicked.connect(lambda: self.select_table(17))
        self.masa19.clicked.connect(lambda: self.select_table(18))
        self.masa20.clicked.connect(lambda: self.select_table(19))

    """Hangi masa seçildiyse yapılması gereken işlemlerin belirlendiği fonksiyon"""
    def select_table(self,capture_id):
        # Kamera numarası belirleniyor
        global cap_id
        cap_id=capture_id
        # set_enabled fonksiyonu çağrılıyor. Masa seçildiğinde nesneler aktif hale geliyor.
        self.set_enabled(True)

        # Sol kısımda masa isminin yazdığı kısma masa adı yazılıyor.
        self.label_17.setText(f'Masa {capture_id+1}')
        # Girdilerin içerisinde yazı varsa temizleniyor.
        self.clear_entries()
        
        # Masalar arası geçişin yapıldığı kısım
        if self.th.isRunning():
            self.th.terminate()
            time.sleep(2)
            self.th.start()
        self.th.start()

    """Liste kutusunun içeriğini temizleyen buton"""
    def clear_list(self):
        self.listWidget.clear()

    """Metin kuruları ve liste kutusunun içi doluysa temizlenmesini sağlayan fonksiyon"""
    def clear_entries(self):
        for i in self.linedits:
            if not i.text()=="":
                i.clear()
        for i in self.result_labels:
            if i.text()!="":
                i.clear()
            if i.styleSheet()=="background-color: rgb(255, 255, 0);":
                i.setStyleSheet("")
        if self.label_21.styleSheet()=="background-color: rgb(255, 255, 0);":
            self.label_21.setStyleSheet("")
        if self.label_21.text()!="":
            self.label_21.clear()

    """Toplama hesabı bulmamızı sağlayan fonksiyon"""
    def calculate(self):
        self.quantity={}
        for i,line_edit in enumerate(self.linedits):
            self.quantity[self.cook_names[i]]=line_edit.text()
        total=0
        for i,edit in enumerate(self.linedits):
            if self.quantity[self.cook_names[i]]!="":
                result=float(self.prices[self.cook_names[i]])*float(self.quantity[self.cook_names[i]])
                total += result
                self.result_labels[i].setText(str(result))
                self.result_labels[i].setStyleSheet(u"background-color: rgb(255, 255, 0);")
                self.label_21.setText(f'Toplam: {total} ₺')
        self.label_21.setStyleSheet(u"background-color: rgb(255, 255, 0);")
            

    """Gönder butonuna basıldığında yemek sayılarının metin kutularına gönderildiği kısım"""
    def update_entry(self):
        items=[]
        for i in range(self.listWidget.count()):
            items.append(self.listWidget.item(i).text())
        names=[]
        price=[]
        for i in items:
            splited=i.split(' ')
            names.append(splited[0])
            price.append(splited[1])
        
        for i in zip(names,price):
            yc=names.count(self.cook_names[0])
            mc=names.count(self.cook_names[1])
            kpc=names.count(self.cook_names[2])
            ec=names.count(self.cook_names[3])
            kf=names.count(self.cook_names[4])
            ky=names.count(self.cook_names[5])
            ik=names.count(self.cook_names[6])
            tk=names.count(self.cook_names[7])
            bak=names.count(self.cook_names[8])
            sut=names.count(self.cook_names[9])
            ko=names.count(self.cook_names[10])
            ay=names.count(self.cook_names[11])

            if i[0]==self.cook_names[0]: # Burada işlemdeki çorba adının Yayla Çorbası olup olmadığının kontrolü yapılıyor
                self.lineEdit.clear()
                self.lineEdit.setText(str(yc))

            elif i[0]==self.cook_names[1]: # Burada işlemdeki çorba adının Mercimek Çorbası olup olmadığının kontrolü yapılıyor
                self.lineEdit_1.clear()
                self.lineEdit_1.setText(str(mc))

            elif i[0]==self.cook_names[2]: # Burada işlemdeki çorba adının Kelle Paça Çorbası olup olmadığının kontrolü yapılıyor
                self.lineEdit_2.clear()
                self.lineEdit_2.setText(str(kpc))

            elif i[0]==self.cook_names[3]: # Burada işlemdeki çorba adının Ezogelin Çorbası olup olmadığının kontrolü yapılıyor
                self.lineEdit_3.clear()
                self.lineEdit_3.setText(str(ec))

            elif i[0]==self.cook_names[4]: # Burada işlemdeki yemek adının Kuru Fasülye olup olmadığının kontrolü yapılıyor
                self.lineEdit_4.clear()
                self.lineEdit_4.setText(str(kf))

            elif i[0]==self.cook_names[5]: # Burada işlemdeki yemek adının Karnıyarık olup olmadığının kontrolü yapılıyor
                self.lineEdit_5.clear()
                self.lineEdit_5.setText(str(ky))

            elif i[0]==self.cook_names[6]: # Burada işlemdeki yemek adının İzmir Köfte olup olmadığının kontrolü yapılıyor
                self.lineEdit_6.clear()
                self.lineEdit_6.setText(str(ik))

            elif i[0]==self.cook_names[7]: # Burada işlemdeki yemek adının Tas Kebabı olup olmadığının kontrolü yapılıyor
                self.lineEdit_7.clear()
                self.lineEdit_7.setText(str(tk))

            elif i[0]==self.cook_names[8]: # Burada işlemdeki tatlı adının Baklava olup olmadığının kontrolü yapılıyor
                self.lineEdit_8.clear()
                self.lineEdit_8.setText(str(bak))

            elif i[0]==self.cook_names[9]: # Burada işlemdeki tatlı adının Sütlaç olup olmadığının kontrolü yapılıyor
                self.lineEdit_9.clear()
                self.lineEdit_9.setText(str(sut))

            elif i[0]==self.cook_names[10]: # Burada işlemdeki içecek adının Kola olup olmadığının kontrolü yapılıyor
                self.lineEdit_10.clear()
                self.lineEdit_10.setText(str(ko))

            elif i[0]==self.cook_names[11]: # Burada işlemdeki içecek adının Ayran olup olmadığının kontrolü yapılıyor
                self.lineEdit_11.clear()
                self.lineEdit_11.setText(str(ay))

    """Thread üzerinden gelen işlenmiş görüntünün ekranda gösterilmesini sağlayan fonksiyon"""
    def process_frame(self,cv_img):
        self.ImageView.view.setXRange(0, cv_img.shape[0])
        self.ImageView.view.setYRange(0, cv_img.shape[1])
        self.ImageView.imageItem.setImage(cv_img)
    
    """
    Thread üzerinden gelen algılanan nesnenin liste kutusuna eklensini sağlayan fonksiyon.
    Burada aynı yemekten birçok isimli yemek varsa onun eklenmemesinin işlemi de yapılmaktadır.
    """
    def add_list(self,detected_object):
        if not self.listWidget.findItems(detected_object,QtCore.Qt.MatchFlag.MatchContains):
            self.listWidget.addItem(detected_object)
                


if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    ra=RestorantApp()
    ra.show()
    app.processEvents()
    sys.exit(app.exec_())
