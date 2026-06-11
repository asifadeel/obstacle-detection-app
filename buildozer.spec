[app]
title = Obstacle Detection
package.name = obstacledetection
package.domain = org.adeel
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite
version = 1.0
requirements = python3,kivy,numpy,opencv,tflite-runtime
assets = assets/best_float32.tflite
orientation = portrait
fullscreen = 1
android.permissions = CAMERA,RECORD_AUDIO
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a

[buildozer]
log_level = 2
