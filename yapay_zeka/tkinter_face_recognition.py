import tkinter as tk
from simple_facerec import SimpleFacerec
import cv2
import pyttsx3


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("İzmir Demokrasi Üniversitesi")

        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images("images/")

        self.face_list = []
        self.detected_faces = set()  # Algılanan yüzleri takip etmek için bir küme
        self.create_widgets()

        # Sesli çıktı için pyttsx3 engine
        self.engine = pyttsx3.init()

    def create_widgets(self):
        etiket = tk.Label(root, text="İzmir Demokrasi Üniversitesi Yoklama Sistemi")
        etiket.pack()
        self.start_button = tk.Button(self.root, text="Yoklama Al", command=self.start_recognition)
        self.start_button.pack(pady=10)

        self.end_button = tk.Button(self.root, text="Yoklamayı Bitir", command=self.show_results)
        self.end_button.pack(pady=10)
        etiket = tk.Label(root, text="Yapay Zeka Yoklama Çıktısı")
        etiket.pack()


        self.face_listbox = tk.Listbox(self.root, height=30, width=80)
        self.face_listbox.pack()

    def start_recognition(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            # Görüntü boşsa işlem yapma
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            face_locations, face_names = self.sfr.detect_known_faces(frame)

            if not face_names:
                # Yüz algılanamadığında 'yetkisiz kişi tespit edildi' şeklinde uyarı ver
                self.engine.say("Kişi bulunamadı.")
                self.engine.runAndWait()

            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 238), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 238), 4)

                # Algılanan yüz bilgilerini bir küme üzerinden kontrol ederek ekleyin
                face_id = f"Name: {name}"
                if face_id not in self.detected_faces:
                    self.face_list.append(face_id)
                    self.detected_faces.add(face_id)

                    # Her yüz algılandığında bir kez sesli çıktı al
                    self.engine.say(f"Merhaba {name}")
                    self.engine.runAndWait()

            cv2.imshow("Yuz Tanima Ekrani", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Tkinter penceresini güncelleyin
            self.root.update()
        cap.release()
        cv2.destroyAllWindows()

    def show_results(self):
        self.face_listbox.delete(0, tk.END)
        for face_info in self.face_list:
            self.face_listbox.insert(tk.END, face_info)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
