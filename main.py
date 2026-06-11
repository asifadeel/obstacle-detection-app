
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
from jnius import autoclass
import threading
import time

CLASS_NAMES = [
    "Bike","Building","Car","Person","Stairs","Traffic sign",
    "Electrical Pole","Road","Motorcycle","Dustbin","Dog",
    "Manhole","Tree","Guard rail","Pedestrian crosswalk",
    "Truck","Bus","Bench","Traffic Cone","Fire hydrant",
    "Traffic Barrel","Plant Pot","Electrical Box","Chair","Bicycle Rack"
]

CRITICAL = ["Car","Truck","Bus","Motorcycle","Bike","Stairs",
            "Manhole","Dog","Guard rail","Bicycle Rack"]
MODERATE = ["Person","Traffic Cone","Traffic Barrel","Bench",
            "Fire hydrant","Dustbin","Plant Pot","Chair","Electrical Box"]

WARNINGS = {
    "en": {"critical":"Warning! {} ahead!","moderate":"Caution! {} on your path!","other":"{} detected nearby."},
    "ur": {"critical":"خبردار! {} آگے ہے!","moderate":"احتیاط! {} راستے میں ہے!","other":"{} قریب میں ہے۔"}
}

class ObstacleApp(App):
    def build(self):
        self.language = "en"
        self.confidence_threshold = 0.65
        self.cooldown = 10
        self.last_spoken = {}
        self.running = True

        self.interpreter = tflite.Interpreter(model_path="assets/best_float32.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        TTSClass = autoclass("android.speech.tts.TextToSpeech")
        self.tts = TTSClass(autoclass("org.kivy.android.PythonActivity").mActivity, None)

        layout = BoxLayout(orientation="vertical")
        self.cam = Camera(resolution=(640,480), play=True)
        layout.add_widget(self.cam)

        self.status = Label(text="Point camera at obstacles", size_hint=(1,0.1), font_size="16sp")
        layout.add_widget(self.status)

        btn_row = BoxLayout(size_hint=(1,0.1))
        btn_lang = Button(text="EN / اردو")
        btn_lang.bind(on_press=self.toggle_language)
        btn_row.add_widget(btn_lang)
        btn_stop = Button(text="Stop")
        btn_stop.bind(on_press=self.stop_app)
        btn_row.add_widget(btn_stop)
        layout.add_widget(btn_row)

        Clock.schedule_interval(self.detect, 0.5)
        return layout

    def toggle_language(self, instance):
        self.language = "ur" if self.language == "en" else "en"
        self.status.text = "Language: " + ("Urdu" if self.language == "ur" else "English")

    def should_speak(self, label):
        now = time.time()
        if label not in self.last_spoken:
            self.last_spoken[label] = 0
        if now - self.last_spoken[label] > self.cooldown:
            self.last_spoken[label] = now
            return True
        return False

    def get_warning(self, label):
        w = WARNINGS[self.language]
        if label in CRITICAL: return w["critical"].format(label)
        elif label in MODERATE: return w["moderate"].format(label)
        else: return w["other"].format(label)

    def speak(self, text):
        def run():
            Locale = autoclass("java.util.Locale")
            lang = Locale("ur","PK") if self.language == "ur" else Locale.ENGLISH
            self.tts.setLanguage(lang)
            self.tts.speak(text, autoclass("android.speech.tts.TextToSpeech").QUEUE_FLUSH, None, None)
        threading.Thread(target=run, daemon=True).start()

    def detect(self, dt):
        if not self.running: return
        try:
            texture = self.cam.texture
            if texture is None: return
            frame = np.frombuffer(texture.pixels, dtype=np.uint8)
            frame = frame.reshape(texture.height, texture.width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            frame = cv2.resize(frame, (640,640))
            frame = cv2.flip(frame, 0)
            inp = frame.astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)
            self.interpreter.set_tensor(self.input_details[0]["index"], inp)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            detections = []
            for i in range(output.shape[1]):
                scores = output[4:, i]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence >= self.confidence_threshold:
                    label = CLASS_NAMES[class_id]
                    if self.should_speak(label):
                        detections.append(label)
            if detections:
                critical_found = [l for l in detections if l in CRITICAL]
                moderate_found = [l for l in detections if l in MODERATE]
                top = critical_found[0] if critical_found else (moderate_found[0] if moderate_found else detections[0])
                warning = self.get_warning(top)
                self.status.text = warning
                self.speak(warning)
        except Exception as e:
            self.status.text = "Error: " + str(e)

    def stop_app(self, instance):
        self.running = False
        self.tts.shutdown()
        App.get_running_app().stop()

if __name__ == "__main__":
    ObstacleApp().run()
